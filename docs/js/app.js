/* =========================================================================
   AgroForecast-DE — Dashboard application
   3-column resizable layout · continuous gradient colormaps · dark mode
   ========================================================================= */

(() => {
    'use strict';

    /* ---------- Metadata ---------- */

    const METRIC_LABELS = {
        pred_q50: 'Predicted yield',
        anomaly: 'Anomaly vs. historical',
        anomaly_pct: 'Anomaly',
        ci_width_pct: 'Forecast uncertainty (CI)',
    };

    const METRIC_UNITS = {
        pred_q50: 't/ha',
        anomaly: 't/ha',
        anomaly_pct: '%',
        ci_width_pct: '%',
    };

    // Perceptually-uniform continuous palettes (multi-stop, interpolated in RGB)
    const PALETTES = {
        // Viridis — sequential, perceptually uniform; great for yield magnitude
        viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
        // Custom yellow-green-blue for yield, more "agronomic" feel
        ylgnbu: ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'],
        // RdYlGn — diverging anomaly (red=bad, green=good)
        rdylgn: ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
        // Inferno — sequential warm for uncertainty
        inferno: ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'],
        // Yellow-Orange-Red — alternative for uncertainty (warmer / less dark)
        ylorrd: ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'],
    };

    const BASEMAPS = [
        { name: 'Carto Light', url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', opts: { subdomains: 'abcd', maxZoom: 19, attribution: '&copy; OpenStreetMap, &copy; CARTO' } },
        { name: 'Carto Voyager', url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', opts: { subdomains: 'abcd', maxZoom: 19, attribution: '&copy; OpenStreetMap, &copy; CARTO' } },
        { name: 'Stamen Terrain', url: 'https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png', opts: { maxZoom: 18, attribution: '&copy; Stadia Maps, Stamen Design, OpenStreetMap' } },
    ];

    const LS_KEY = {
        theme: 'agro-theme',
        leftW: 'agro-left-w',
        rightW: 'agro-right-w',
        leftOpen: 'agro-left-open',
        rightOpen: 'agro-right-open',
        cardOrder: 'agro-card-order',
        cardCollapsed: 'agro-card-collapsed',
    };

    const state = {
        index: [],
        crop: null,
        date: null,
        metric: 'pred_q50',
        level: 'district',
        stateFilter: '',
        district: null,
        stateData: null,
        pinned: null,
        basemapIdx: 0,
        theme: 'light',
        charts: {},
        scale: null,
    };

    /* ---------- Boot ---------- */

    document.addEventListener('DOMContentLoaded', init);

    function init() {
        initTheme();
        restoreLayout();
        const map = initMap();
        bindUI(map);
        initResize();
        initCards();

        fetch('data/forecast_index.json')
            .then((r) => r.json())
            .then((idx) => {
                state.index = idx;
                populateFilters();
                loadActive(map);
            })
            .catch((err) => console.error('Failed to load forecast_index.json', err));
    }

    /* ---------- Theme ---------- */

    function initTheme() {
        const stored = localStorage.getItem(LS_KEY.theme);
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        state.theme = stored || (prefersDark ? 'dark' : 'light');
        document.body.dataset.theme = state.theme;
    }
    function toggleTheme() {
        state.theme = state.theme === 'dark' ? 'light' : 'dark';
        document.body.dataset.theme = state.theme;
        localStorage.setItem(LS_KEY.theme, state.theme);
        rerenderCharts();
    }
    function themeColors() {
        const css = getComputedStyle(document.body);
        return {
            text: css.getPropertyValue('--text').trim() || '#0f172a',
            muted: css.getPropertyValue('--text-muted').trim() || '#64748b',
            border: css.getPropertyValue('--border').trim() || '#e2e8f0',
            accent: css.getPropertyValue('--accent').trim() || '#16a34a',
            bg: css.getPropertyValue('--bg-elev').trim() || '#ffffff',
        };
    }

    /* ---------- Layout / panels ---------- */

    function restoreLayout() {
        const root = document.documentElement;
        const lw = parseInt(localStorage.getItem(LS_KEY.leftW), 10);
        const rw = parseInt(localStorage.getItem(LS_KEY.rightW), 10);
        if (Number.isFinite(lw) && lw >= 220 && lw <= 600) root.style.setProperty('--col-left-w', `${lw}px`);
        if (Number.isFinite(rw) && rw >= 240 && rw <= 720) root.style.setProperty('--col-right-w', `${rw}px`);

        const ws = document.getElementById('workspace');
        if (localStorage.getItem(LS_KEY.leftOpen) === '0') ws.classList.add('no-left');
        if (localStorage.getItem(LS_KEY.rightOpen) === '0') ws.classList.add('no-right');
        updateTogglesState();
    }
    function updateTogglesState() {
        const ws = document.getElementById('workspace');
        document.getElementById('toggle-left').classList.toggle('is-active', !ws.classList.contains('no-left'));
        document.getElementById('toggle-right').classList.toggle('is-active', !ws.classList.contains('no-right'));
    }

    function initResize() {
        const handles = document.querySelectorAll('.resize-handle');
        handles.forEach((h) => {
            h.addEventListener('mousedown', (e) => startResize(e, h));
            h.addEventListener('touchstart', (e) => startResize(e.touches[0], h, true), { passive: true });
        });
    }
    function startResize(evtOrTouch, handle, isTouch = false) {
        const target = handle.dataset.target;
        const root = document.documentElement;
        const startX = evtOrTouch.clientX;
        const startLeft = parseFloat(getComputedStyle(root).getPropertyValue('--col-left-w'));
        const startRight = parseFloat(getComputedStyle(root).getPropertyValue('--col-right-w'));
        handle.classList.add('is-active');
        document.body.classList.add('is-resizing');

        const move = (e) => {
            const x = (e.touches ? e.touches[0] : e).clientX;
            const dx = x - startX;
            if (target === 'left') {
                const w = Math.min(600, Math.max(220, startLeft + dx));
                root.style.setProperty('--col-left-w', `${w}px`);
            } else {
                const w = Math.min(720, Math.max(240, startRight - dx));
                root.style.setProperty('--col-right-w', `${w}px`);
            }
            resizeAllCharts();
            if (mapRef) mapRef.invalidateSize();
        };
        const stop = () => {
            handle.classList.remove('is-active');
            document.body.classList.remove('is-resizing');
            document.removeEventListener(isTouch ? 'touchmove' : 'mousemove', move);
            document.removeEventListener(isTouch ? 'touchend' : 'mouseup', stop);
            localStorage.setItem(LS_KEY.leftW, parseInt(getComputedStyle(root).getPropertyValue('--col-left-w'), 10));
            localStorage.setItem(LS_KEY.rightW, parseInt(getComputedStyle(root).getPropertyValue('--col-right-w'), 10));
        };
        document.addEventListener(isTouch ? 'touchmove' : 'mousemove', move, { passive: false });
        document.addEventListener(isTouch ? 'touchend' : 'mouseup', stop);
    }

    /* ---------- Cards: drag-reorder + collapse ---------- */

    function initCards() {
        const stack = document.getElementById('card-stack');

        // Restore order
        const savedOrder = JSON.parse(localStorage.getItem(LS_KEY.cardOrder) || 'null');
        if (Array.isArray(savedOrder)) {
            savedOrder.forEach((key) => {
                const el = stack.querySelector(`[data-card="${key}"]`);
                if (el) stack.appendChild(el);
            });
        }

        // Restore collapsed state
        const collapsed = JSON.parse(localStorage.getItem(LS_KEY.cardCollapsed) || '{}');
        stack.querySelectorAll('.card').forEach((card) => {
            if (collapsed[card.dataset.card]) card.classList.add('is-collapsed');
        });

        // Collapse buttons
        stack.querySelectorAll('.card-collapse').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const card = btn.closest('.card');
                card.classList.toggle('is-collapsed');
                const c = JSON.parse(localStorage.getItem(LS_KEY.cardCollapsed) || '{}');
                c[card.dataset.card] = card.classList.contains('is-collapsed');
                localStorage.setItem(LS_KEY.cardCollapsed, JSON.stringify(c));
                requestAnimationFrame(resizeAllCharts);
            });
        });

        // Make header double-click also toggle collapse
        stack.querySelectorAll('.card-head').forEach((head) => {
            head.addEventListener('dblclick', () => {
                head.querySelector('.card-collapse').click();
            });
        });

        // Drag-to-reorder
        let dragSrc = null;
        stack.querySelectorAll('.card').forEach((card) => {
            card.addEventListener('dragstart', (e) => {
                dragSrc = card;
                card.classList.add('is-dragging');
                e.dataTransfer.effectAllowed = 'move';
                try { e.dataTransfer.setData('text/plain', card.dataset.card); } catch (_) { /* ignore */ }
            });
            card.addEventListener('dragend', () => {
                card.classList.remove('is-dragging');
                stack.querySelectorAll('.card').forEach((c) => c.classList.remove('drag-over'));
                saveCardOrder();
            });
            card.addEventListener('dragover', (e) => {
                e.preventDefault();
                if (!dragSrc || dragSrc === card) return;
                card.classList.add('drag-over');
                const rect = card.getBoundingClientRect();
                const before = (e.clientY - rect.top) < rect.height / 2;
                if (before) stack.insertBefore(dragSrc, card);
                else stack.insertBefore(dragSrc, card.nextSibling);
            });
            card.addEventListener('dragleave', () => card.classList.remove('drag-over'));
            card.addEventListener('drop', (e) => { e.preventDefault(); card.classList.remove('drag-over'); });
        });
    }
    function saveCardOrder() {
        const order = Array.from(document.querySelectorAll('.card-stack .card')).map((c) => c.dataset.card);
        localStorage.setItem(LS_KEY.cardOrder, JSON.stringify(order));
    }

    /* ---------- Map ---------- */

    let mapRef, baseLayer, geoLayer;

    function initMap() {
        mapRef = L.map('map', { zoomControl: true, attributionControl: true, preferCanvas: true })
            .setView([51.2, 10.45], 6);
        applyBasemap();
        return mapRef;
    }
    function applyBasemap() {
        const bm = BASEMAPS[state.basemapIdx];
        if (baseLayer) mapRef.removeLayer(baseLayer);
        baseLayer = L.tileLayer(bm.url, bm.opts).addTo(mapRef);
    }

    /* ---------- UI bindings ---------- */

    function bindUI(map) {
        document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
        document.getElementById('basemap-toggle').addEventListener('click', () => {
            state.basemapIdx = (state.basemapIdx + 1) % BASEMAPS.length;
            applyBasemap();
        });
        document.getElementById('reset-view').addEventListener('click', () => {
            map.setView([51.2, 10.45], 6);
            state.pinned = null;
            renderRegionDetail(null);
            renderLayer();
        });

        // Panel toggles
        document.getElementById('toggle-left').addEventListener('click', () => {
            const ws = document.getElementById('workspace');
            ws.classList.toggle('no-left');
            localStorage.setItem(LS_KEY.leftOpen, ws.classList.contains('no-left') ? '0' : '1');
            updateTogglesState();
            setTimeout(() => { resizeAllCharts(); mapRef && mapRef.invalidateSize(); }, 50);
        });
        document.getElementById('toggle-right').addEventListener('click', () => {
            const ws = document.getElementById('workspace');
            ws.classList.toggle('no-right');
            localStorage.setItem(LS_KEY.rightOpen, ws.classList.contains('no-right') ? '0' : '1');
            updateTogglesState();
            setTimeout(() => { resizeAllCharts(); mapRef && mapRef.invalidateSize(); }, 50);
        });

        // Level toggle
        document.querySelectorAll('.seg-btn[data-level]').forEach((btn) => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.seg-btn[data-level]').forEach((b) => b.classList.remove('is-active'));
                btn.classList.add('is-active');
                state.level = btn.dataset.level;
                document.getElementById('overlay-level').textContent = state.level === 'state' ? 'States' : 'Districts';
                refreshAll();
            });
        });

        // Selectors
        document.getElementById('crop-select').addEventListener('change', (e) => { state.crop = e.target.value; loadActive(map); });
        document.getElementById('forecast-date').addEventListener('change', (e) => { state.date = e.target.value; loadActive(map); });
        document.getElementById('metric-select').addEventListener('change', (e) => {
            state.metric = e.target.value;
            refreshAll();
        });
        document.getElementById('state-filter').addEventListener('change', (e) => {
            state.stateFilter = e.target.value;
            refreshAll();
            zoomToFiltered();
        });

        // Search
        const search = document.getElementById('search-input');
        const results = document.getElementById('search-results');
        search.addEventListener('input', () => updateSearchResults(search.value, results));
        search.addEventListener('focus', () => updateSearchResults(search.value, results));
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.control')) results.hidden = true;
        });

        window.addEventListener('resize', () => { resizeAllCharts(); mapRef && mapRef.invalidateSize(); });
    }

    function refreshAll() {
        state.scale = null;
        renderLayer();
        updateLegend();
        renderCharts();
        renderRegionDetail(state.pinned);
    }

    /* ---------- Filters ---------- */

    function populateFilters() {
        const crops = [...new Set(state.index.map((d) => d.crop))].sort();
        const dates = [...new Set(state.index.map((d) => d.date))].sort().reverse();
        const cropSel = document.getElementById('crop-select');
        const dateSel = document.getElementById('forecast-date');
        cropSel.innerHTML = crops.map((c) => `<option value="${c}">${prettyCrop(c)}</option>`).join('');
        dateSel.innerHTML = dates.map((d) => `<option value="${d}">${prettyDate(d)}</option>`).join('');
        state.crop = state.crop || crops[0];
        state.date = state.date || dates[0];
        cropSel.value = state.crop;
        dateSel.value = state.date;
    }
    function prettyCrop(c) { return c.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()); }
    function prettyDate(d) {
        const [y, m] = d.split('-');
        const month = new Date(`${y}-${m}-01`).toLocaleString('en', { month: 'long' });
        return `${month} ${y}`;
    }

    /* ---------- Data ---------- */

    function loadActive(map) {
        const entry = state.index.find((d) => d.crop === state.crop && d.date === state.date);
        if (!entry) return;
        const distP = fetch(`data/${entry.file}`).then((r) => r.json());
        const stateP = entry.state_file ? fetch(`data/${entry.state_file}`).then((r) => r.json()) : Promise.resolve(null);
        Promise.all([distP, stateP]).then(([d, s]) => {
            state.district = d;
            state.stateData = s;
            populateStateFilter();
            updateOverlayMeta();
            updateKpis();
            refreshAll();
        });
    }

    function populateStateFilter() {
        const states = [...new Set(state.district.features.map((f) => f.properties.NUTS1_ID))]
            .map((id) => ({
                id,
                name: state.district.features.find((f) => f.properties.NUTS1_ID === id).properties.NUTS1_NAME,
            }))
            .sort((a, b) => a.name.localeCompare(b.name));
        const sel = document.getElementById('state-filter');
        const prev = state.stateFilter;
        sel.innerHTML = `<option value="">All states</option>` +
            states.map((s) => `<option value="${s.id}">${s.name}</option>`).join('');
        sel.value = prev;
    }
    function updateOverlayMeta() {
        document.getElementById('overlay-crop').textContent = prettyCrop(state.crop);
        document.getElementById('overlay-date').textContent = prettyDate(state.date);
        document.getElementById('overlay-level').textContent = state.level === 'state' ? 'States' : 'Districts';
    }

    /* ---------- Active features ---------- */

    function activeFeatures() {
        if (state.level === 'state') return state.stateData ? state.stateData.features : [];
        const all = state.district ? state.district.features : [];
        return state.stateFilter ? all.filter((f) => f.properties.NUTS1_ID === state.stateFilter) : all;
    }
    function activeGeoJson() {
        if (state.level === 'state') return state.stateData;
        if (!state.district) return null;
        if (!state.stateFilter) return state.district;
        return { type: 'FeatureCollection', features: state.district.features.filter((f) => f.properties.NUTS1_ID === state.stateFilter) };
    }

    /* ---------- Color: continuous interpolation ---------- */

    function hex2rgb(h) {
        const n = parseInt(h.slice(1), 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    function rgb2css(r, g, b) { return `rgb(${r|0}, ${g|0}, ${b|0})`; }

    function interpolatePalette(t, palette) {
        if (!Number.isFinite(t)) return '#9ca3af';
        t = Math.max(0, Math.min(1, t));
        const n = palette.length - 1;
        const idx = t * n;
        const i0 = Math.floor(idx);
        const i1 = Math.min(n, i0 + 1);
        const f = idx - i0;
        const [r0, g0, b0] = hex2rgb(palette[i0]);
        const [r1, g1, b1] = hex2rgb(palette[i1]);
        return rgb2css(r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f);
    }

    function paletteFor(metric) {
        if (metric === 'anomaly' || metric === 'anomaly_pct') return PALETTES.rdylgn;
        if (metric === 'ci_width_pct') return PALETTES.ylorrd;
        return PALETTES.viridis;
    }

    function makeScale(metric) {
        // Use 2nd–98th percentile for the visible range so outliers don't flatten the gradient
        const vals = activeFeatures().map((f) => f.properties[metric]).filter(Number.isFinite);
        if (!vals.length) return { palette: paletteFor(metric), domain: [0, 1], type: 'seq', metric };
        const sorted = vals.slice().sort((a, b) => a - b);
        const pick = (p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))];
        let lo = pick(0.02);
        let hi = pick(0.98);
        if (lo === hi) { lo -= 0.5; hi += 0.5; }

        const palette = paletteFor(metric);
        if (metric === 'anomaly' || metric === 'anomaly_pct') {
            const m = Math.max(Math.abs(lo), Math.abs(hi), 0.1);
            return { palette, domain: [-m, m], type: 'div', metric, ticks: [-m, -m / 2, 0, m / 2, m] };
        }
        return {
            palette,
            domain: [lo, hi],
            type: 'seq',
            metric,
            ticks: [lo, lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.5, lo + (hi - lo) * 0.75, hi],
        };
    }

    function colorFor(value, scale) {
        if (!Number.isFinite(value)) return '#9ca3af';
        const [lo, hi] = scale.domain;
        const t = (value - lo) / Math.max(hi - lo, 1e-9);
        return interpolatePalette(t, scale.palette);
    }

    function getScale() {
        if (!state.scale || state.scale.metric !== state.metric) state.scale = makeScale(state.metric);
        return state.scale;
    }

    /* ---------- Layer ---------- */

    function renderLayer() {
        if (!mapRef) return;
        if (geoLayer) { mapRef.removeLayer(geoLayer); geoLayer = null; }
        const data = activeGeoJson();
        if (!data || !data.features.length) return;
        state.scale = makeScale(state.metric);

        geoLayer = L.geoJson(data, {
            renderer: L.canvas(),
            style: (f) => featureStyle(f, state.scale),
            onEachFeature,
        }).addTo(mapRef);

        updateLegend();
    }

    function featureStyle(f, scale) {
        const v = f.properties[state.metric];
        const isPinned = state.pinned && (
            (state.level === 'state' && f.properties.NUTS1_ID === state.pinned.NUTS1_ID && state.pinned._level === 'state') ||
            (state.level !== 'state' && f.properties.district_id === state.pinned.district_id)
        );
        return {
            fillColor: colorFor(v, scale),
            fillOpacity: 0.88,
            weight: isPinned ? 2.5 : (state.level === 'state' ? 1.2 : 0.55),
            color: isPinned ? (state.theme === 'dark' ? '#fff' : '#0f172a') : (state.level === 'state' ? '#475569' : '#94a3b8'),
        };
    }

    function onEachFeature(feature, layer) {
        layer.on({
            mouseover: (e) => {
                const t = e.target;
                t.setStyle({ weight: 2.5, color: state.theme === 'dark' ? '#fff' : '#0f172a' });
                t.bringToFront();
                renderRegionDetail({ ...feature.properties, _level: state.level });
            },
            mouseout: () => {
                if (geoLayer) geoLayer.resetStyle(layer);
                renderRegionDetail(state.pinned);
            },
            click: (e) => {
                state.pinned = { ...feature.properties, _level: state.level };
                renderRegionDetail(state.pinned);
                renderLayer();
                mapRef.fitBounds(e.target.getBounds(), { padding: [40, 40], maxZoom: 9 });
            },
        });
    }

    function zoomToFiltered() {
        if (!geoLayer) return;
        try {
            const bounds = geoLayer.getBounds();
            if (bounds.isValid()) mapRef.fitBounds(bounds, { padding: [30, 30] });
        } catch (e) { /* ignore */ }
    }

    /* ---------- Legend (continuous gradient) ---------- */

    function updateLegend() {
        const el = document.getElementById('legend');
        if (!el) return;
        const scale = getScale();
        const palette = scale.palette;
        const unit = METRIC_UNITS[state.metric] || '';
        const fmt = (v) => {
            if (!Number.isFinite(v)) return '—';
            const abs = Math.abs(v);
            if (abs >= 100) return v.toFixed(0);
            if (abs >= 10) return v.toFixed(1);
            return v.toFixed(2);
        };

        // CSS gradient string: sample palette into n stops so the gradient is smooth
        const stops = palette.map((c, i) => `${c} ${((i / (palette.length - 1)) * 100).toFixed(1)}%`).join(', ');
        const [lo, hi] = scale.domain;
        const ticks = scale.ticks || [lo, (lo + hi) / 2, hi];

        el.innerHTML = `
            <div class="legend-title">${METRIC_LABELS[state.metric]}${unit ? ` (${unit})` : ''}</div>
            <div class="legend-gradient" style="background: linear-gradient(to right, ${stops});"></div>
            <div class="legend-ticks">
                ${ticks.map((t) => {
                    const pos = ((t - lo) / Math.max(hi - lo, 1e-9)) * 100;
                    return `<span class="legend-tick" style="left:${Math.max(0, Math.min(100, pos))}%">${fmt(t)}</span>`;
                }).join('')}
            </div>
        `;
    }

    /* ---------- KPIs ---------- */

    function updateKpis() {
        const grid = document.getElementById('kpi-grid');
        const feats = state.district ? state.district.features : [];
        if (!feats.length) { grid.innerHTML = ''; return; }
        const yields = feats.map((f) => f.properties.pred_q50).filter(Number.isFinite);
        const anoms = feats.map((f) => f.properties.anomaly_pct).filter(Number.isFinite);
        const cis = feats.map((f) => f.properties.ci_width_pct).filter(Number.isFinite);
        const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
        const meanY = mean(yields), meanA = mean(anoms), meanCI = mean(cis);
        const positiveShare = (anoms.filter((a) => a > 0).length / anoms.length) * 100;
        grid.innerHTML = `
            <div class="kpi">
                <div class="kpi-label">Mean yield</div>
                <div class="kpi-value">${meanY.toFixed(1)} <span class="kpi-sub">t/ha</span></div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Mean anomaly</div>
                <div class="kpi-value ${meanA >= 0 ? 'kpi-pos' : 'kpi-neg'}">
                    ${meanA >= 0 ? '+' : ''}${meanA.toFixed(1)}<span class="kpi-sub"> %</span>
                </div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Above normal</div>
                <div class="kpi-value">${positiveShare.toFixed(0)}<span class="kpi-sub"> %</span></div>
                <div class="kpi-sub">of ${anoms.length} districts</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Mean CI width</div>
                <div class="kpi-value">${meanCI.toFixed(1)}<span class="kpi-sub"> %</span></div>
            </div>
        `;
    }

    /* ---------- Region detail ---------- */

    function renderRegionDetail(props) {
        const host = document.getElementById('region-detail');
        if (!props) {
            host.innerHTML = `<p class="placeholder">Hover or click a region on the map to see its forecast detail.</p>`;
            return;
        }
        const isState = props._level === 'state';
        const name = isState ? props.NUTS1_NAME : props.NUTS3_NAME;
        const stateLine = isState ? `Federal state · ${props.NUTS1_ID}` : `${props.NUTS1_NAME} · ${props.district_id}`;
        const ensSpread = (props.pred_q90 != null && props.pred_q10 != null) ? props.pred_q90 - props.pred_q10 : null;

        const items = [
            { label: 'Predicted (q50)', value: `${fmtNum(props.pred_q50)} t/ha`, sub: `q10 ${fmtNum(props.pred_q10)} – q90 ${fmtNum(props.pred_q90)}` },
            { label: 'Anomaly', value: `${signed(props.anomaly)} t/ha`, sub: `${signed(props.anomaly_pct)} %`, cls: numCls(props.anomaly) },
            { label: 'Historical mean', value: `${fmtNum(props.hist_mean)} t/ha`, sub: `Ref: ${props.ref_period || '—'}` },
        ];
        if (!isState) {
            items.push({ label: 'Uncertainty', value: `±${fmtNum((props.ci_width || 0) / 2)} t/ha`, sub: `CI ${fmtNum(props.ci_width_pct)}%` });
            items.push({ label: 'Ensemble', value: `${props.n_members || 50} members`, sub: `min ${fmtNum(props.ens_min)} · max ${fmtNum(props.ens_max)}` });
        }

        const min = props.ens_min ?? props.pred_q10;
        const max = props.ens_max ?? props.pred_q90;
        const lo = props.pred_q10, hi = props.pred_q90, med = props.pred_q50, hist = props.hist_mean;
        const allVals = [min, max, lo, hi, med, hist].filter(Number.isFinite);
        const range = allVals.length ? [Math.min(...allVals), Math.max(...allVals)] : [0, 1];
        const span = Math.max(range[1] - range[0], 0.001);
        const pct = (v) => Math.max(0, Math.min(100, ((v - range[0]) / span) * 100));

        const ensembleBar = (lo != null && hi != null) ? `
            <div class="ens-bar-wrap">
                <div class="ens-bar-label">
                    <span>Ensemble q10–q90${ensSpread != null ? ` · span ${fmtNum(ensSpread)} t/ha` : ''}</span>
                    <span>${fmtNum(range[0])} – ${fmtNum(range[1])} t/ha</span>
                </div>
                <div class="ens-bar">
                    <div class="ens-fill" style="left:${pct(lo)}%; right:${100 - pct(hi)}%"></div>
                    ${Number.isFinite(med) ? `<div class="ens-tick" style="left:calc(${pct(med)}% - 1px)" title="q50"></div>` : ''}
                    ${Number.isFinite(hist) ? `<div class="ens-tick hist" style="left:calc(${pct(hist)}% - 1px)" title="historical mean"></div>` : ''}
                </div>
            </div>` : '';

        host.innerHTML = `
            <div class="region-header">
                <div>
                    <h3>${name}</h3>
                    <div class="region-state">${stateLine}</div>
                </div>
                <div class="region-pin ${state.pinned ? 'pinned' : ''}">${state.pinned ? 'Pinned' : 'Hovering'}</div>
            </div>
            <div class="region-grid">
                ${items.map((it) => `
                    <div class="region-item">
                        <div class="label">${it.label}</div>
                        <div class="value ${it.cls || ''}">${it.value}</div>
                        ${it.sub ? `<div class="sub">${it.sub}</div>` : ''}
                    </div>
                `).join('')}
            </div>
            ${ensembleBar}
        `;
    }

    function fmtNum(v) {
        if (v == null || !Number.isFinite(v)) return '—';
        return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
    }
    function signed(v) {
        if (v == null || !Number.isFinite(v)) return '—';
        return (v >= 0 ? '+' : '') + (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
    }
    function numCls(v) {
        if (v == null || !Number.isFinite(v) || v === 0) return '';
        return v > 0 ? 'kpi-pos' : 'kpi-neg';
    }

    /* ---------- Charts ---------- */

    function renderCharts() {
        renderDistribution();
        renderTopBottom();
        renderStatesChart();
    }
    function rerenderCharts() {
        Object.values(state.charts).forEach((c) => c && c.destroy());
        state.charts = {};
        renderCharts();
        updateLegend();
    }
    function resizeAllCharts() {
        Object.values(state.charts).forEach((c) => c && c.resize());
    }
    function chartCommonOptions() {
        const c = themeColors();
        Chart.defaults.color = c.muted;
        Chart.defaults.borderColor = c.border;
        Chart.defaults.font.family = "'Inter', sans-serif";
        return c;
    }

    function renderDistribution() {
        const c = chartCommonOptions();
        const ctx = document.getElementById('chart-distribution');
        if (!ctx) return;
        const feats = activeFeatures();
        if (!feats.length) { if (state.charts.dist) { state.charts.dist.destroy(); delete state.charts.dist; } return; }
        const vals = feats.map((f) => f.properties[state.metric]).filter(Number.isFinite);
        if (!vals.length) return;
        const min = Math.min(...vals), max = Math.max(...vals);
        const nBins = Math.min(24, Math.max(10, Math.round(Math.sqrt(vals.length))));
        const binW = (max - min) / nBins || 1;
        const counts = new Array(nBins).fill(0);
        const labels = [];
        const centers = [];
        for (let i = 0; i < nBins; i++) {
            const lo = min + i * binW, hi = lo + binW;
            labels.push(lo.toFixed(1));
            centers.push(lo + binW / 2);
            for (const v of vals) if (v >= lo && (v < hi || (i === nBins - 1 && v <= hi))) counts[i]++;
        }
        const scale = getScale();
        const colors = centers.map((v) => colorFor(v, scale));

        if (state.charts.dist) state.charts.dist.destroy();
        state.charts.dist = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets: [{ data: counts, backgroundColor: colors, borderColor: 'transparent', borderRadius: 2, barPercentage: 1.0, categoryPercentage: 0.98 }] },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                animation: { duration: 250 },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { title: (i) => `${i[0].label} ${METRIC_UNITS[state.metric] || ''}` } },
                },
                scales: {
                    x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, autoSkipPadding: 8 }, title: { display: true, text: `${METRIC_LABELS[state.metric]} (${METRIC_UNITS[state.metric] || ''})`, color: c.muted } },
                    y: { beginAtZero: true, title: { display: true, text: `# ${state.level === 'state' ? 'states' : 'districts'}`, color: c.muted }, grid: { color: c.border } },
                },
            },
        });
    }

    function renderTopBottom() {
        const c = chartCommonOptions();
        const ctx = document.getElementById('chart-topbottom');
        if (!ctx) return;
        const feats = activeFeatures().slice();
        if (!feats.length) { if (state.charts.tb) { state.charts.tb.destroy(); delete state.charts.tb; } return; }
        const labelKey = state.level === 'state' ? 'NUTS1_NAME' : 'NUTS3_NAME';
        feats.sort((a, b) => (b.properties[state.metric] ?? -Infinity) - (a.properties[state.metric] ?? -Infinity));
        const top = feats.slice(0, 10);
        const bottom = feats.slice(-10).reverse();
        const items = [...top, ...bottom];
        const labels = items.map((f) => f.properties[labelKey]);
        const data = items.map((f) => f.properties[state.metric]);
        const scale = getScale();
        const colors = data.map((v) => colorFor(v, scale));

        if (state.charts.tb) state.charts.tb.destroy();
        state.charts.tb = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: 'transparent', borderRadius: 3 }] },
            options: {
                indexAxis: 'y',
                maintainAspectRatio: false,
                responsive: true,
                animation: { duration: 250 },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: (i) => ` ${i.parsed.x?.toFixed(2)} ${METRIC_UNITS[state.metric] || ''}` } },
                },
                scales: {
                    x: { grid: { color: c.border }, title: { display: true, text: METRIC_UNITS[state.metric] || '', color: c.muted } },
                    y: { ticks: { autoSkip: false, font: { size: 10 } }, grid: { display: false } },
                },
            },
        });
    }

    function renderStatesChart() {
        const c = chartCommonOptions();
        const ctx = document.getElementById('chart-states');
        if (!ctx || !state.stateData) return;
        const feats = state.stateData.features.slice().sort((a, b) => b.properties.pred_q50 - a.properties.pred_q50);
        const labels = feats.map((f) => f.properties.NUTS1_NAME);
        const med = feats.map((f) => f.properties.pred_q50);
        const lo = feats.map((f) => f.properties.pred_q10);
        const hi = feats.map((f) => f.properties.pred_q90);
        const hist = feats.map((f) => f.properties.hist_mean);

        if (state.charts.states) state.charts.states.destroy();
        state.charts.states = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    { label: 'q10–q90 range', data: feats.map((f, i) => [lo[i], hi[i]]), backgroundColor: c.accent + '33', borderColor: c.accent, borderWidth: 1, borderSkipped: false, borderRadius: 4, order: 2 },
                    { type: 'line', label: 'q50 forecast', data: med, borderColor: c.accent, backgroundColor: c.accent, pointRadius: 4, pointHoverRadius: 6, showLine: false, order: 1 },
                    { type: 'line', label: 'Historical mean', data: hist, borderColor: c.muted, backgroundColor: c.muted, pointRadius: 3, pointStyle: 'rectRot', showLine: false, order: 0 },
                ],
            },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                animation: { duration: 250 },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 10, color: c.muted, font: { size: 10 } } },
                    tooltip: { mode: 'index', intersect: false },
                },
                scales: {
                    x: { ticks: { font: { size: 9 }, maxRotation: 60, minRotation: 40 }, grid: { display: false } },
                    y: { beginAtZero: false, title: { display: true, text: 't/ha', color: c.muted }, grid: { color: c.border } },
                },
            },
        });
    }

    /* ---------- Search ---------- */

    function updateSearchResults(q, host) {
        if (!state.district) return;
        host.innerHTML = '';
        host.hidden = true;
        if (!q || q.length < 2) return;
        const needle = q.toLowerCase();
        const hits = state.district.features
            .filter((f) => f.properties.NUTS3_NAME.toLowerCase().includes(needle) || f.properties.district_id.toLowerCase().includes(needle))
            .slice(0, 12);
        if (!hits.length) return;
        hits.forEach((f) => {
            const li = document.createElement('li');
            li.innerHTML = `${f.properties.NUTS3_NAME}<span class="sr-sub">${f.properties.NUTS1_NAME} · ${f.properties.district_id}</span>`;
            li.addEventListener('mousedown', (e) => {
                e.preventDefault();
                state.pinned = { ...f.properties, _level: 'district' };
                if (state.level !== 'district') {
                    document.querySelectorAll('.seg-btn[data-level]').forEach((b) => b.classList.toggle('is-active', b.dataset.level === 'district'));
                    state.level = 'district';
                    document.getElementById('overlay-level').textContent = 'Districts';
                    refreshAll();
                } else {
                    renderRegionDetail(state.pinned);
                    renderLayer();
                }
                const tmp = L.geoJson(f);
                mapRef.fitBounds(tmp.getBounds(), { padding: [40, 40], maxZoom: 10 });
                host.hidden = true;
                document.getElementById('search-input').value = '';
            });
            host.appendChild(li);
        });
        host.hidden = false;
    }
})();
