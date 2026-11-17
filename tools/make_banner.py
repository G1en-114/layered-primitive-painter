"""Generate assets/banner.svg: layered Gaussian strokes forming 'strata' behind the title."""
import math
import random
import sys

W, H = 1280, 360
rng = random.Random(114)

# bottom -> top strata: (center y, spread, palette, count, size range, opacity)
LAYERS = [
    (300, 70, ["#1b2a6b", "#23408e", "#2a5bb5"], 46, (60, 150), 0.75),
    (235, 55, ["#1f7a8c", "#2aa198", "#3fb6a8"], 44, (40, 110), 0.75),
    (170, 50, ["#e0823d", "#f2a541", "#d9643a"], 40, (28, 85), 0.75),
    (105, 45, ["#e85d9b", "#c84b9e", "#f07ab8"], 34, (18, 60), 0.70),
    (60, 40, ["#fbe7a1", "#ffffff", "#ffd6e8"], 26, (8, 28), 0.70),
]

defs, strokes = [], []
gid = 0
for cy, spread, pal, n, (smin, smax), op in LAYERS:
    for _ in range(n):
        x = rng.uniform(-40, W + 40)
        # strata bend gently across the width
        y = cy + 18 * math.sin(x / W * math.pi * 1.6 + cy / 50) + rng.gauss(0, spread * 0.35)
        sx = rng.uniform(smin, smax)
        sy = sx * rng.uniform(0.18, 0.55)
        ang = rng.gauss(-8, 14)
        col = rng.choice(pal)
        hard = rng.random()  # edge hardness beta: soft airbrush vs. solid dab
        inner = 0.15 + 0.6 * hard
        a = op * rng.uniform(0.6, 1.0)
        defs.append(
            f'<radialGradient id="g{gid}"><stop offset="0" stop-color="{col}" stop-opacity="{a:.2f}"/>'
            f'<stop offset="{inner:.2f}" stop-color="{col}" stop-opacity="{a * 0.85:.2f}"/>'
            f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></radialGradient>'
        )
        strokes.append(
            f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{sx:.1f}" ry="{sy:.1f}" fill="url(#g{gid})" '
            f'transform="rotate({ang:.1f} {x:.1f} {y:.1f})"/>'
        )
        gid += 1

FONT = "'Inter','SF Pro Display','Segoe UI','Helvetica Neue',Helvetica,Arial,sans-serif"

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Strata">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0b1020"/><stop offset="1" stop-color="#131a3a"/></linearGradient>
<linearGradient id="title" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ffffff"/><stop offset="0.55" stop-color="#ffe3c2"/><stop offset="1" stop-color="#ffb3d1"/></linearGradient>
<radialGradient id="veil" cx="0.5" cy="0.5" r="0.5" gradientTransform="translate(0.5 0.5) scale(1 2.2) translate(-0.5 -0.5)"><stop offset="0" stop-color="#0b1020" stop-opacity="0.7"/><stop offset="0.6" stop-color="#0b1020" stop-opacity="0.25"/><stop offset="1" stop-color="#0b1020" stop-opacity="0"/></radialGradient>
<clipPath id="clip"><rect width="{W}" height="{H}" rx="22"/></clipPath>
<filter id="soft" x="-5%" y="-5%" width="110%" height="110%"><feGaussianBlur stdDeviation="0.8"/></filter>
<filter id="glow" x="-10%" y="-30%" width="120%" height="160%"><feGaussianBlur in="SourceAlpha" stdDeviation="10" result="b"/><feFlood flood-color="#000" flood-opacity="0.55"/><feComposite in2="b" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>
{"".join(defs)}
</defs>
<g clip-path="url(#clip)">
<rect width="{W}" height="{H}" fill="url(#bg)"/>
<g filter="url(#soft)">{"".join(strokes)}</g>
<rect width="{W}" height="{H}" fill="url(#veil)"/>
<g filter="url(#glow)">
<text x="{W / 2}" y="{H / 2}" text-anchor="middle" dominant-baseline="central" font-family="{FONT}" font-size="170" font-weight="800" letter-spacing="-5" fill="url(#title)">Strata</text>
</g>
</g>
</svg>
'''
open(sys.argv[1], "w").write(svg)
print(gid, "strokes", len(svg) // 1024, "KiB")
