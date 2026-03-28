#!/usr/bin/env python3
"""Build standalone HTML slides for RSA predictions analysis (CSP style)."""

import os

ANALYZE_DIR = os.path.join(os.path.dirname(__file__), "..", "analyze")

def read_b64(filename):
    with open(os.path.join(ANALYZE_DIR, filename + ".b64"), "r") as f:
        return f.read().strip()

img_conditions = read_b64("rsa_vs_human_conditions.png")
img_balance    = read_b64("rsa_info_balance.png")
img_stacked    = read_b64("rsa_stacked_all_conditions.png")
img_match      = read_b64("rsa_match_nonmatch.png")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSA Predictions Explain the Uniform Info Distribution</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{
    --glaucous: #7581B3; --shimmer: #C65353; --crayola: #E2BA78;
    --fern: #5C7457; --opal: #99C2C2; --independence: #575463;
    --bg: #FAFAFA; --text: #2D2D2D;
  }}
  body {{ font-family: 'Source Sans 3', Arial, sans-serif; color: var(--text); background: #000; }}
  .slide {{
    width: 1280px; height: 720px; background: var(--bg);
    margin: 0 auto 4px; padding: 50px 70px;
    position: relative; overflow: hidden; page-break-after: always;
  }}
  .slide.title-slide {{
    background: var(--independence); color: white;
    display: flex; flex-direction: column; justify-content: center; padding: 60px 80px;
  }}
  .slide.title-slide h1 {{ font-size: 42px; font-weight: 700; line-height: 1.2; margin-bottom: 16px; }}
  .slide.title-slide .subtitle {{ font-size: 22px; font-weight: 300; color: var(--opal); margin-bottom: 40px; }}
  .slide.title-slide .meta {{ font-size: 16px; color: #aaa; line-height: 1.6; }}
  .slide.title-slide .accent-bar {{ position: absolute; top: 0; left: 0; width: 8px; height: 100%; background: var(--crayola); }}
  .slide h2 {{ font-size: 28px; font-weight: 700; color: var(--independence); margin-bottom: 20px; border-bottom: 3px solid var(--glaucous); padding-bottom: 8px; }}
  .slide .two-col {{ display: flex; gap: 36px; height: calc(100% - 70px); }}
  .slide .col {{ flex: 1; }}
  .slide .col-wide {{ flex: 1.4; }}
  .slide .col-narrow {{ flex: 0.7; }}
  .slide ul {{ list-style: none; padding: 0; }}
  .slide ul li {{ font-size: 17px; line-height: 1.5; padding: 5px 0 5px 18px; position: relative; }}
  .slide ul li::before {{ content: ''; position: absolute; left: 0; top: 13px; width: 7px; height: 7px; border-radius: 50%; background: var(--glaucous); }}
  .slide .highlight {{ background: rgba(117,129,179,0.1); border-left: 4px solid var(--glaucous); padding: 10px 14px; margin: 10px 0; font-size: 16px; }}
  .slide img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  .slide .figure-container {{ display: flex; justify-content: center; align-items: center; height: calc(100% - 70px); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 16px; margin: 8px 0; }}
  th {{ background: var(--independence); color: white; padding: 8px 12px; text-align: left; font-weight: 600; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  tr.best {{ background: #e8f5e9; font-weight: 600; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; margin: 1px 3px; }}
  .tag.high {{ background: var(--crayola); color: #333; }}
  .tag.low {{ background: var(--glaucous); color: white; }}
  .tag.info {{ background: var(--fern); color: white; }}
  .tag.good {{ background: #d4edda; color: #155724; }}
  .tag.bad {{ background: #f8d7da; color: #721c24; }}
  .tag.best {{ background: #c8e6c9; color: #1b5e20; }}
  .slide-num {{ position: absolute; bottom: 18px; right: 28px; font-size: 12px; color: #aaa; }}
  .footnote {{ position: absolute; bottom: 18px; left: 70px; font-size: 11px; color: #999; }}
  .slide.section-slide {{
    background: var(--glaucous); color: white;
    display: flex; flex-direction: column; justify-content: center; padding: 60px 80px;
  }}
  .slide.section-slide h2 {{ font-size: 42px; font-weight: 700; border: none; color: white; margin-bottom: 16px; }}
  .slide.section-slide p {{ font-size: 20px; color: rgba(255,255,255,0.7); }}
  .slide.section-slide .section-num {{
    font-size: 120px; font-weight: 700; color: rgba(255,255,255,0.1);
    position: absolute; right: 80px; bottom: 40px;
  }}
  .stat-row {{ display: flex; gap: 20px; margin: 12px 0; }}
  .stat-card {{
    flex: 1; background: white; border: 2px solid #e0e0e0;
    border-radius: 8px; padding: 14px 16px; text-align: center;
  }}
  .stat-card .label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-card .value {{ font-size: 36px; font-weight: 700; margin: 4px 0; }}
  .stat-card .note {{ font-size: 12px; color: #999; }}
  .stat-card.high {{ border-color: var(--crayola); }}
  .stat-card.high .value {{ color: var(--crayola); }}
  .stat-card.low {{ border-color: var(--glaucous); }}
  .stat-card.low .value {{ color: var(--glaucous); }}
  .stat-card.info {{ border-color: var(--fern); }}
  .stat-card.info .value {{ color: var(--fern); }}
  @media print {{ body {{ background: white; }} .slide {{ margin: 0; box-shadow: none; }} }}
</style>
</head>
<body>

<!-- SLIDE 1: Title -->
<div class="slide title-slide">
  <div class="accent-bar"></div>
  <h1>Why Listeners Guess Uniformly<br>for Informative Speakers</h1>
  <div class="subtitle">RSA predictions explain the near-uniform human response distribution in the info condition</div>
  <div class="meta">LLMarg Project &mdash; RSA vs Human Comparison</div>
  <span class="slide-num">1/7</span>
</div>

<!-- SLIDE 2: The Puzzle -->
<div class="slide">
  <h2>The Puzzle: Why Is the Info Condition So Flat?</h2>
  <div class="two-col">
    <div class="col">
      <p style="font-size:15px;margin-bottom:12px;">When listeners hear an <strong>informative</strong> speaker, their responses are near-uniform across all three categories &mdash; close to chance (1/3):</p>
      <div class="stat-row">
        <div class="stat-card high">
          <div class="label">P(high)</div>
          <div class="value">32.5%</div>
          <div class="note">Non-match</div>
        </div>
        <div class="stat-card info">
          <div class="label">P(info)</div>
          <div class="value">36.0%</div>
          <div class="note">Match</div>
        </div>
        <div class="stat-card low">
          <div class="label">P(low)</div>
          <div class="value">31.5%</div>
          <div class="note">Non-match</div>
        </div>
      </div>
      <div class="highlight" style="margin-top:14px;">
        <strong>Question:</strong> Why is the informative condition so flat when <span class="tag high">high</span> (63.3%) and <span class="tag low">low</span> (54.4%) speakers are identified well above chance?
      </div>
    </div>
    <div class="col" style="display:flex;flex-direction:column;justify-content:center;">
      <p style="font-size:15px;font-weight:700;color:var(--independence);margin-bottom:10px;">Contrast with High &amp; Low</p>
      <table style="font-size:15px;">
        <tr><th>Condition</th><th>Match</th><th>P(high)</th><th>P(info)</th><th>P(low)</th></tr>
        <tr><td><span class="tag high">high</span></td><td><strong>63.3%</strong></td><td style="color:var(--crayola);font-weight:700;">63.3%</td><td>19.7%</td><td>17.0%</td></tr>
        <tr><td><span class="tag info">info</span></td><td><strong>36.3%</strong></td><td>32.5%</td><td style="color:var(--fern);font-weight:700;">36.0%</td><td>31.5%</td></tr>
        <tr><td><span class="tag low">low</span></td><td><strong>54.4%</strong></td><td>15.6%</td><td>30.0%</td><td style="color:var(--glaucous);font-weight:700;">54.4%</td></tr>
      </table>
      <div class="highlight" style="margin-top:12px;font-size:14px;">
        <strong>Key observation:</strong> In the info row, P(high) &asymp; P(low). The two non-match responses are <em>balanced</em>.
      </div>
    </div>
  </div>
  <span class="slide-num">2/7</span>
</div>

<!-- SLIDE 3: RSA Prediction Structure -->
<div class="slide">
  <h2>RSA Predictions: The Key Mechanism</h2>
  <div class="two-col">
    <div class="col-narrow" style="display:flex;flex-direction:column;justify-content:center;">
      <p style="font-size:15px;font-weight:700;color:var(--independence);margin-bottom:10px;">RSA (nonparametric) vs Human</p>
      <table style="font-size:15px;">
        <tr><th></th><th>P(high)</th><th>P(info)</th><th>P(low)</th></tr>
        <tr><td><strong>RSA</strong></td><td style="color:var(--shimmer);font-weight:600;">20.8%</td><td>58.4%</td><td style="color:var(--glaucous);font-weight:600;">20.8%</td></tr>
        <tr><td><strong>Human</strong></td><td style="color:var(--shimmer);font-weight:600;">32.5%</td><td>36.0%</td><td style="color:var(--glaucous);font-weight:600;">31.5%</td></tr>
      </table>
      <div class="highlight" style="margin-top:14px;font-size:14px;">
        <strong>Crucial:</strong> P(high) &asymp; P(low) in <em>both</em> RSA and human data.<br><br>
        The informative speaker gives no directional evidence, so the two framing responses are <strong>perfectly balanced</strong>.
      </div>
      <div class="highlight" style="font-size:14px;border-left-color:var(--shimmer);">
        RSA is more confident in <span class="tag info">info</span> (58% vs 36%), but the <em>symmetry</em> between non-matches is the same.
      </div>
    </div>
    <div class="col-wide" style="display:flex;align-items:center;justify-content:center;">
      <img src="data:image/png;base64,{img_conditions}" style="max-height:520px;">
    </div>
  </div>
  <span class="slide-num">3/7</span>
</div>

<!-- SLIDE 4: Item-Level Balance -->
<div class="slide">
  <h2>Item-Level Symmetry in Info Predictions</h2>
  <div class="two-col">
    <div class="col" style="display:flex;align-items:center;justify-content:center;">
      <img src="data:image/png;base64,{img_balance}" style="max-height:500px;">
    </div>
    <div class="col" style="display:flex;flex-direction:column;justify-content:center;">
      <p style="font-size:15px;margin-bottom:10px;">For each of the 10 informative items, the RSA model predicts <strong>P(high) &asymp; P(low)</strong>.</p>
      <p style="font-size:15px;margin-bottom:14px;">Items come in <strong>mirror pairs</strong> (1&amp;2, 3&amp;4, &hellip;) where quantifiers swap. This creates exact symmetry: what one item assigns to P(high), its mirror assigns to P(low).</p>
      <div class="stat-row">
        <div class="stat-card" style="border-color:var(--independence);">
          <div class="label">Mean |P(high) &minus; P(low)|</div>
          <div class="value" style="font-size:28px;color:var(--independence);">0.167</div>
          <div class="note">per-item difference</div>
        </div>
        <div class="stat-card" style="border-color:var(--shimmer);">
          <div class="label">Aggregate</div>
          <div class="value" style="font-size:28px;color:var(--shimmer);">P(h) = P(l)</div>
          <div class="note">20.8% each</div>
        </div>
      </div>
      <div class="highlight" style="margin-top:10px;font-size:14px;">
        Individual items may favor one side, but across items they <strong>cancel out perfectly</strong>.
      </div>
    </div>
  </div>
  <span class="slide-num">4/7</span>
</div>

<!-- SLIDE 5: All Three Conditions -->
<div class="slide">
  <h2>The Full Picture: RSA Predictions Across All Conditions</h2>
  <div class="figure-container">
    <img src="data:image/png;base64,{img_stacked}" style="max-height:520px;">
  </div>
  <div class="footnote"><strong>High/Low:</strong> RSA strongly predicts the matching response (~85%). <strong>Info:</strong> Mass splits between P(info) and balanced P(high)/P(low).</div>
  <span class="slide-num">5/7</span>
</div>

<!-- SLIDE 6: Match vs Non-Match -->
<div class="slide">
  <h2>Match vs Non-Match: RSA Explains the Asymmetry</h2>
  <div class="figure-container" style="flex-direction:column;">
    <img src="data:image/png;base64,{img_match}" style="max-height:420px;">
    <div style="margin-top:12px;max-width:1000px;">
      <ul>
        <li>For <span class="tag high">high</span> and <span class="tag low">low</span> conditions, RSA predicts a <strong>dominant match response</strong> (~85%) + negligible opposite (~0.6%)</li>
        <li>For <span class="tag info">info</span>, both non-match responses are <strong>equally small</strong> &mdash; no directional bias whatsoever</li>
        <li>Human data shows the same qualitative pattern, with less extreme values (match = 36% for info vs RSA's 58%)</li>
      </ul>
    </div>
  </div>
  <span class="slide-num">6/7</span>
</div>

<!-- SLIDE 7: Takeaway -->
<div class="slide section-slide">
  <h2>Key Takeaway</h2>
  <p style="font-size:22px;color:rgba(255,255,255,0.9);max-width:900px;line-height:1.6;">
    The near-uniform human distribution in the info condition is <strong>not random noise</strong> &mdash; it is a <em>predicted consequence</em> of rational pragmatic inference.
  </p>
  <p style="margin-top:20px;font-size:18px;color:rgba(255,255,255,0.65);max-width:800px;line-height:1.5;">
    When the speaker provides no argumentative direction, the RSA model predicts P(high) = P(low), leaving listeners with no basis to prefer one framing over the other. This balanced non-match prediction + moderate match probability yields the observed near-uniform pattern.
  </p>
  <div class="section-num">?</div>
  <span class="slide-num" style="color:rgba(255,255,255,0.3);">7/7</span>
</div>

</body>
</html>"""

output_path = os.path.join(os.path.dirname(__file__), "rsa_info_explanation.html")
with open(output_path, "w") as f:
    f.write(html)
print(f"Written to {output_path}")
