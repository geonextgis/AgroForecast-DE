# Dashboard Logo

The dashboard logo shown in the top-left of the header is **auto-loaded** from
this folder. To change it, simply drop your image file here.

## How it works

`docs/index.html` contains an `<img>` inside the `.brand-mark` element that
points at `assets/logo.svg`:

```html
<div class="brand-mark" aria-hidden="true">
    <img class="brand-logo"
         src="assets/logo.svg"
         alt=""
         onload="this.parentElement.classList.add('has-logo')"
         onerror="this.remove()">
    <svg class="brand-logo-fallback" ...>…default leaf icon…</svg>
</div>
```

- If `assets/logo.svg` **loads successfully**, the `<img>` is shown and the
  green gradient background of the mark is removed (`.has-logo` class).
- If it **fails to load** (file missing), the `<img>` is silently removed and
  the default inline-SVG leaf icon is shown instead.

## Steps to change the logo

1. **Replace** (or add) the file:
   ```
   docs/assets/logo.svg
   ```
   SVG is recommended for crispness at any size, but a square PNG (≥128×128)
   also works.

2. If your file uses a different name or extension, update the `src` in
   `docs/index.html`:
   ```html
   <img class="brand-logo" src="assets/your-logo.png" ...>
   ```

3. **Optional styling.** The container is 36×36 px. If your logo looks too
   small/large or needs padding, override these CSS rules in
   `docs/css/style.css`:
   ```css
   .brand-mark { width: 40px; height: 40px; }       /* enlarge container */
   .brand-logo { padding: 2px; }                    /* inner spacing */
   .brand-mark.has-logo { background: #fff; }       /* keep a background */
   ```

4. **Hard refresh** (Ctrl/Cmd-Shift-R) — browsers cache images aggressively.

## Notes

- This folder is also the right place for any other static brand assets
  (favicons, partner logos used elsewhere on the page, etc.).
- The `<title>` text and brand name shown next to the logo are set in
  `docs/index.html` (`<title>` tag and `.brand-text h1`).
