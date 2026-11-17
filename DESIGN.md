---
name: Farmland VLM
description: A seed-packet shell for reading aerial farmland tiles; flat field green, packet yellow selection, tomato state stamps.
colors:
  field-green: "#14532d"
  field-green-hi: "#1d6a3b"
  meadow-ground: "#e9f1e4"
  packet-white: "#ffffff"
  seedling-wash: "#f3f8ef"
  furrow-line: "#cfe0c6"
  soil-ink: "#10281a"
  leaf-ink: "#3d5a47"
  moss-muted: "#4f6858"
  packet-yellow: "#f5b700"
  tomato-stamp: "#c8401b"
  error-red: "#b3261e"
  rail-text: "#f3f9ee"
  rail-note: "#c9e0cf"
  track-green: "#d9e6d0"
rounded:
  stamp: "3px"
  control: "6px"
  card: "8px"
  pill: "50%"
spacing:
  xs: "6px"
  sm: "10px"
  md: "12px"
  lg: "16px"
  xl: "24px"
typography:
  display:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 800
    lineHeight: 1.2
  title:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 800
    lineHeight: 1.15
  body:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.45
  stamp:
    fontFamily: "Archivo Narrow, Archivo, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0.08em"
components:
  rail:
    backgroundColor: "{colors.field-green}"
    textColor: "{colors.rail-text}"
    padding: "22px 14px"
    width: "232px"
  rail-tool:
    backgroundColor: "transparent"
    textColor: "{colors.rail-text}"
    rounded: "{rounded.control}"
    padding: "12px"
  rail-tool-hover:
    backgroundColor: "{colors.field-green-hi}"
  rail-tool-on:
    backgroundColor: "{colors.packet-yellow}"
    textColor: "{colors.soil-ink}"
  button-outline:
    backgroundColor: "{colors.packet-white}"
    textColor: "{colors.soil-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "6px 14px"
  button-outline-hover:
    backgroundColor: "{colors.seedling-wash}"
  button-outline-on:
    backgroundColor: "{colors.field-green}"
    textColor: "#ffffff"
  button-send:
    backgroundColor: "{colors.field-green}"
    textColor: "#ffffff"
    rounded: "{rounded.control}"
    width: "42px"
  stamp:
    backgroundColor: "transparent"
    textColor: "{colors.tomato-stamp}"
    typography: "{typography.stamp}"
    rounded: "{rounded.stamp}"
    padding: "2px 8px"
  stamp-ok:
    backgroundColor: "{colors.field-green}"
    textColor: "#ffffff"
  stamp-info:
    backgroundColor: "transparent"
    textColor: "{colors.field-green}"
  reading-card:
    backgroundColor: "{colors.packet-white}"
    rounded: "{rounded.card}"
    padding: "16px"
    width: "400px"
  text-field:
    backgroundColor: "{colors.packet-white}"
    textColor: "{colors.soil-ink}"
    rounded: "{rounded.control}"
    padding: "9px 12px"
  tile-frame:
    backgroundColor: "{colors.seedling-wash}"
    rounded: "{rounded.card}"
  thumb:
    backgroundColor: "{colors.packet-white}"
    rounded: "{rounded.control}"
    size: "76px"
---

# Design System: Farmland VLM

## Overview

**Creative North Star: "The Seed Packet"**

Every reading is a labelled packet: plain facts, a stamped state, no guessing. The shell is a committed field green; the working surfaces are pale green-tinted paper and white cards held by 2px outlines; selection is packet yellow; model state is a tomato rubber stamp. Colour is flat. There is no glass, no decorative gradient and no soft ambient glow except on one floating search dropdown.

The system is built for a demo audience who judge the model in a few minutes, so it is dense, legible and honest: state and tile source are named in stamps, uncertainty is shown as a hedge stamp plus a confidence bar, and the two tools (Aerial reader, Field map) each own a full view so they are never mistaken for one linked product.

**Key Characteristics:**
- Fixed 232px green rail on desktop, collapsing to a two-column top bar at 880px and below.
- Flat colour with 2px outlines on controls and cards, 3px on tile and map frames.
- Archivo for everything, Archivo Narrow caps only for stamps, brand and tile labels.
- One accent per job: yellow selects, tomato stamps, green commits.
- The 3x3 plot grid is the recurring signature geometry (tile overlay, drop zone, finding glyph).

## Colors

A green-committed palette on a tinted ground, with yellow and tomato each confined to a single role.

### Primary
- **Field Green** (`{colors.field-green}`): rail background, outlines of controls and frames, filled active segments, send button, confidence fill.
- **Field Green Hi** (`{colors.field-green-hi}`): hover on rail tools and send button only.

### Secondary
- **Packet Yellow** (`{colors.packet-yellow}`): selection only: active rail tool, active map tool, selected thumbnail border, text selection, the located-cell hint, the focus ring on the rail.
- **Tomato Stamp** (`{colors.tomato-stamp}`): state stamps (hedged) and the emphasised word in a finding name. Deliberate token decision: shipped as #c8401b, not the sketch's #e4572e, because 12px stamp text on white needs about 4.9:1 contrast.

### Neutral
- **Meadow Ground** (`{colors.meadow-ground}`): page background.
- **Packet White** (`{colors.packet-white}`): cards, inputs, buttons at rest.
- **Seedling Wash** (`{colors.seedling-wash}`): hover fill, tile and drop-zone base.
- **Furrow Line** (`{colors.furrow-line}`): 1px dividers and 2px card borders, drop-zone grid lines.
- **Track Green** (`{colors.track-green}`): empty tracks of confidence, extent bars and 3x3 glyph cells.
- **Soil Ink** (`{colors.soil-ink}`): primary text and focus ring. **Leaf Ink** (`{colors.leaf-ink}`): secondary text. **Moss Muted** (`{colors.moss-muted}`): captions and placeholders.
- **Error Red** (`{colors.error-red}`): errors and out-of-range box size.
- **Rail Text** and **Rail Note**: text on the green rail.

### Named Rules
**The Stamp Colour Rule.** Tomato is for model state and tile source only. It never decorates, fills a button or marks a link.

**The One Yellow Rule.** Yellow means "this is the selected thing". If nothing is selected, no yellow appears except the brand mark's first cell.

**The Data Colours Are Not The Palette Rule.** Anomaly-class colours (`scheme.ts`), the NDVI ramp, rain blue and map layer colours are data encodings; do not reuse them for chrome.

## Typography

**Display and Body Font:** Archivo (with system-ui, sans-serif)
**Stamp Font:** Archivo Narrow (with Archivo, sans-serif), caps, tracked 0.08em

**Character:** A plain, slightly industrial grotesque at heavy weight for answers and a condensed caps voice for labels that behave like printed stamps.

### Hierarchy
- **Display** (800, 24px, 1.2): the answer headline in the reading card (22px inside paired answers, 20px when muted).
- **Title** (800, 22px, 1.15): view title; panel headings 18px.
- **Body** (400, 15px, 1.45): base; captions at 15.5px / 1.5; summary 16px.
- **Label** (600, 14px): buttons, segments, model select; 13px for compact secondary controls.
- **Stamp** (700, 12px, 0.08em, uppercase): state stamps; brand 20px; tile labels 11px.
- Small text: 13px, 12.5px and 12px for meta and keys. Numbers that update use tabular figures.

### Named Rules
**The Narrow Caps Rule.** Archivo Narrow caps are reserved for stamps, the brand and tile labels. Running text and buttons stay in Archivo sentence case.

## Layout

Desktop is a two-column grid: 232px rail, then a fluid main view. Each view is a column with a header row (title, scope sentence, controls) and a body split into a stage (flexible) and a 400px side card (340px under 1100px). Views fill the viewport height and scroll internally; at 880px and below the page scrolls, the rail becomes a top bar with two tool buttons, the body stacks, the tile is square at viewport width (max 520px) and the map is 56vh (min 320px). Spacing is a small hand-set rhythm: 6, 8, 10, 12, 16, 20, 24px; view padding 20px 24px 24px. The aerial tile is always a square sized by the container.

## Elevation & Depth

Flat by default. Depth is conveyed by outline weight and tonal steps (ground, white card, wash), not shadow. The single shadow is the search results dropdown (`0 8px 20px rgba(16,40,26,0.18)`), which floats over the map. Selected thumbnails use a double ring (yellow border plus a 2px green ring), not a shadow. Focus is a 3px ink outline with 2px offset, yellow on the rail.

### Named Rules
**The Flat-By-Default Rule.** Surfaces carry no shadow. A shadow is only for a layer that overlaps the map.

## Shapes

Small, firm corners: stamps 3px, controls 6px, cards, tiles and map frame 8px, status dots and pulse circles 50%. Borders are always visible: 2px on controls and cards, 3px on tile, map and drop-zone frames (the drop zone is dashed). Stamps sit rotated -2deg and land with a short scale-in; this is the only tilted element.

## Components

### Rail and Tools
Green column with the grid-mark brand, two tool buttons (12px padding, 6px radius), a model-status note pinned to the bottom. Hover lifts to Field Green Hi; the active tool is packet yellow with ink text and a small dot.

### Buttons
- **Outline:** white fill, 2px green border, 6px radius, 6px 14px padding, 14px/600. Hover fills Seedling Wash; pressed moves 1px down; "on" fills green with white text.
- **Segmented:** a joined row inside one 2px green border; the active segment fills green; disabled options are muted and struck through.
- **Send:** 42px green square-ish button with white icon; hover Field Green Hi.
- **Suggestion link:** underlined green text button, used for suggested questions.

### Stamps
Outlined 12px Archivo Narrow caps in tomato (default, hedged). Variants: ok (filled green, white text, "did not hedge") and info (green outline, e.g. Idle, Nothing found).

### Reading Card
White card, 2px Furrow Line border, 8px radius. Entries are separated by 1px lines; each holds the question, headline, stamp, confidence bar, caption and findings rows. The compose bar sits at the bottom behind a 2px divider. Findings rows pair a 3x3 plot glyph, name, extent bars and a confidence mini-bar.

### Inputs
White, 2px green border, 6px radius, 9px 12px padding; placeholder in Moss Muted; disabled at 55% opacity. Focus uses the global ink outline.

### Tile Frame and Plot Grid
Square image in a 3px green frame, 8px radius, with a 3x3 overlay of hairline white cells that light up for findings and a dark pill label top-left. The empty drop zone repeats the same 3x3 grid in Furrow Line on a dashed 3px frame.

### Thumbnail Strip
76px square thumbs, 2px Furrow Line border; hover green; selected yellow with green ring; upload tile is dashed green.

### Field Panel
White side card with summary, season chart (ink line, dashed blue wet-days line, translucent rain bars), layer toggles and a red-to-green NDVI legend. The map frame is 3px green; map controls take the 2px green outline with no shadow.

## Do's and Don'ts

### Do:
- **Do** keep colour flat with 2px outlines (3px on tile/map frames) and rely on tonal steps for depth.
- **Do** use packet yellow only for selection and tomato only for stamps.
- **Do** label the source and state of each reading with a stamp, and keep hedge plus confidence visible.
- **Do** draw the 3x3 plot grid whenever a tile or a finding location is shown.
- **Do** respect `prefers-reduced-motion`: stamp landing, pulse and caret animations stop and transitions are disabled.
- **Do** keep text at AA contrast; the tomato token was darkened for exactly this.

### Don't:
- **Don't** link or visually blend Aerial reader and Field map; the drawn box is never an input to the chat.
- **Don't** add glass, decorative gradients, or ambient glows.
- **Don't** reuse anomaly or NDVI data colours as interface colours.
- **Don't** use Archivo Narrow caps for body copy or buttons.
