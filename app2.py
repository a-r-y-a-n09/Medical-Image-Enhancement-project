"""
╔══════════════════════════════════════════════════════════╗
║   MEDVISION AI - Medical Image Enhancement System       ║
║   Auto-optimize · Claude Vision · Full Pipeline         ║
╚══════════════════════════════════════════════════════════╝

Install:  pip install streamlit opencv-python pillow numpy plotly requests
Run:      streamlit run medical_image_enhancer.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io, zipfile, base64, requests
import plotly.graph_objects as go

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(page_title="MedVision AI", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
.main{background:#080c14;}
.block-container{padding:1.5rem 2rem 3rem 2rem;max-width:1400px;}
section[data-testid="stSidebar"]{background:#070a10;border-right:1px solid #1a2035;}
section[data-testid="stSidebar"] *{font-family:'Sora',sans-serif !important;}

.hero{background:linear-gradient(135deg,#0a1628 0%,#0d1f3c 50%,#091522 100%);
  border:1px solid #1e3a5f;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;
  position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-50%;right:-10%;width:400px;height:400px;
  background:radial-gradient(circle,rgba(0,180,255,0.06) 0%,transparent 70%);pointer-events:none;}
.hero-title{font-size:2rem;font-weight:700;color:#e8f4ff;margin:0 0 0.3rem 0;letter-spacing:-0.02em;}
.hero-sub{font-size:0.9rem;color:#4a7a9b;font-weight:300;letter-spacing:0.08em;
  text-transform:uppercase;font-family:'DM Mono',monospace;}
.hero-badges{margin-top:1rem;display:flex;gap:8px;flex-wrap:wrap;}
.badge{background:rgba(0,140,255,0.1);border:1px solid rgba(0,140,255,0.25);color:#5bb8ff;
  border-radius:20px;padding:3px 12px;font-size:0.72rem;font-family:'DM Mono',monospace;letter-spacing:0.05em;}
.badge.green{background:rgba(0,200,120,0.08);border-color:rgba(0,200,120,0.25);color:#40d9a0;}
.badge.amber{background:rgba(255,180,0,0.08);border-color:rgba(255,180,0,0.25);color:#ffc43d;}

.metrics-row{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:1.5rem;}
.metric-card{background:#0c1220;border:1px solid #1a2840;border-radius:12px;
  padding:1rem 1.1rem;position:relative;overflow:hidden;}
.metric-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#0070cc,#00b8ff);border-radius:12px 12px 0 0;}
.metric-card.good::after{background:linear-gradient(90deg,#00a060,#40d9a0);}
.metric-card.warn::after{background:linear-gradient(90deg,#cc7000,#ffc43d);}
.metric-label{font-size:0.68rem;color:#3d5a78;letter-spacing:0.1em;text-transform:uppercase;
  font-family:'DM Mono',monospace;margin-bottom:0.4rem;}
.metric-value{font-size:1.6rem;font-weight:700;color:#c8e0ff;line-height:1;font-family:'DM Mono',monospace;}
.metric-unit{font-size:0.75rem;color:#3d5a78;margin-left:4px;}
.metric-sub{font-size:0.7rem;color:#2a4060;margin-top:0.3rem;font-family:'DM Mono',monospace;}

.section-title{font-size:0.72rem;font-weight:600;color:#2a5a8a;letter-spacing:0.15em;
  text-transform:uppercase;font-family:'DM Mono',monospace;margin:1.8rem 0 0.8rem 0;
  display:flex;align-items:center;gap:8px;}
.section-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1a2840,transparent);}

.auto-info{background:linear-gradient(135deg,#081a10,#091520);border:1px solid #1a4030;
  border-radius:10px;padding:0.8rem 1rem;font-size:0.8rem;color:#40c090;
  font-family:'DM Mono',monospace;margin-bottom:1rem;line-height:1.6;}

.ai-insights{background:linear-gradient(135deg,#080f20,#0a1828,#080f1a);
  border:1px solid #1a3060;border-radius:14px;padding:1.8rem 2rem;
  line-height:1.85;color:#b8d4f0;font-size:0.9rem;}
.ai-header{font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#2a5a9a;
  font-family:'DM Mono',monospace;margin-bottom:1rem;padding-bottom:0.6rem;border-bottom:1px solid #1a3060;}
.ai-disclaimer{font-size:0.72rem;color:#2a4060;margin-top:1.2rem;padding-top:0.8rem;
  border-top:1px solid #1a2840;font-family:'DM Mono',monospace;}

.diag-table{width:100%;border-collapse:collapse;font-family:'DM Mono',monospace;font-size:0.78rem;}
.diag-table th{color:#2a5070;font-weight:500;text-align:left;padding:0.4rem 0.8rem;
  border-bottom:1px solid #1a2840;letter-spacing:0.08em;text-transform:uppercase;font-size:0.68rem;}
.diag-table td{color:#6090b8;padding:0.5rem 0.8rem;border-bottom:1px solid #0f1a28;}
.diag-table td:first-child{color:#c0d8f0;} .diag-table td:nth-child(2){color:#40c090;}

.sb-section{font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#1e4060;
  font-family:'DM Mono',monospace;padding:0.6rem 0 0.3rem 0;margin-top:0.5rem;border-top:1px solid #0f1e30;}
[data-testid="metric-container"]{display:none;}
</style>
""", unsafe_allow_html=True)

# ─── UTILITIES ──────────────────────────────────────────────

def load_image(f):
    pil = Image.open(f).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return pil, bgr, gray

def image_quality(gray):
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hn = hist/hist.sum()
    entropy = -np.sum(hn[hn>0]*np.log2(hn[hn>0]))
    return {"mean":float(gray.mean()),"std":float(gray.std()),
            "sharpness":float(lap_var),"entropy":float(entropy),
            "min":int(gray.min()),"max":int(gray.max())}

def auto_tune(gray):
    q = image_quality(gray)
    b5 = cv2.GaussianBlur(gray,(5,5),0)
    noise = float(np.abs(gray.astype(float)-b5.astype(float)).mean())
    h,w = gray.shape
    et = cv2.Canny(b5,50,150)
    ed = float(et.sum()/(255*h*w)*100)
    v = float(np.median(gray))
    clow  = max(0,  int(max(0.5*v, v-0.33*q["std"])))
    chigh = min(255,int(min(1.5*v, v+0.66*q["std"])))
    blur_k = 9 if noise>12 else 5 if noise>6 else 3
    clip = 4.0 if q["std"]<30 else 2.5 if q["std"]<55 else 1.5
    sstr = round(min(2.5,max(0.3,2.5-q["std"]/55)),1)
    mk = 5 if ed>3 else 3
    reasons = {
        "Blur Kernel":(blur_k,f"noise≈{noise:.2f}"),
        "CLAHE Clip":(clip,f"std={q['std']:.1f}"),
        "Canny Low":(clow,f"median={v:.0f}, std={q['std']:.1f}"),
        "Canny High":(chigh,f"median={v:.0f}, std={q['std']:.1f}"),
        "Sharpening":(sstr,f"std={q['std']:.1f} → inverse boost"),
        "Morph Kernel":(mk,f"edge_density={ed:.2f}%"),
        "Segmentation":("Otsu","self-calibrating thresholding"),
    }
    return {"blur_k":blur_k,"median_k":3,"clahe_clip":clip,"clahe_tile":8,
            "canny_low":clow,"canny_high":chigh,"sharpen_str":sstr,
            "seg_method":"Otsu","thresh_val":128,"morph_op":"Close","morph_k":mk,
            "_reasons":reasons,"_quality":q,"_noise":round(noise,2),"_ed":round(ed,2)}

def run_pipeline(gray, p):
    blurred  = cv2.GaussianBlur(gray,(p["blur_k"],p["blur_k"]),0)
    denoised = cv2.medianBlur(blurred,p["median_k"])
    clahe    = cv2.createCLAHE(clipLimit=p["clahe_clip"],tileGridSize=(p["clahe_tile"],p["clahe_tile"]))
    enhanced = clahe.apply(denoised)
    if p["sharpen_str"]>0:
        k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=np.float32)*p["sharpen_str"]
        k[1,1]+=1
        sharpened = np.clip(cv2.filter2D(enhanced,-1,k),0,255).astype(np.uint8)
    else:
        sharpened = enhanced
    smooth = cv2.GaussianBlur(sharpened,(3,3),0)
    edges  = cv2.Canny(smooth,p["canny_low"],p["canny_high"])
    seg_fn = {
        "Binary":           lambda: cv2.threshold(sharpened,p["thresh_val"],255,cv2.THRESH_BINARY)[1],
        "Otsu":             lambda: cv2.threshold(sharpened,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
        "Adaptive Mean":    lambda: cv2.adaptiveThreshold(sharpened,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2),
        "Adaptive Gaussian":lambda: cv2.adaptiveThreshold(sharpened,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),
    }
    seg = seg_fn[p["seg_method"]]()
    morph_map = {"None":lambda s:s,
        "Dilate":lambda s:cv2.morphologyEx(s,cv2.MORPH_DILATE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(p["morph_k"],p["morph_k"]))),
        "Erode": lambda s:cv2.morphologyEx(s,cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(p["morph_k"],p["morph_k"]))),
        "Open":  lambda s:cv2.morphologyEx(s,cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(p["morph_k"],p["morph_k"]))),
        "Close": lambda s:cv2.morphologyEx(s,cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(p["morph_k"],p["morph_k"]))),
    }
    return denoised, enhanced, sharpened, edges, morph_map[p["morph_op"]](seg)

def to_b64(img):
    _,buf=cv2.imencode(".png",img)
    return base64.b64encode(buf.tobytes()).decode()

def to_png(img):
    _,buf=cv2.imencode(".png",img)
    return buf.tobytes()

def build_zip(outputs):
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as zf:
        for n,a in outputs.items(): zf.writestr(f"{n}.png",to_png(a))
    return bio.getvalue()

def histogram_fig(g1, g2):
    bins=np.arange(256)
    h1=cv2.calcHist([g1],[0],None,[256],[0,256]).flatten()
    h2=cv2.calcHist([g2],[0],None,[256],[0,256]).flatten()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=bins,y=h1,name="Original",fill="tozeroy",
        line=dict(color="#2266aa",width=1.5),fillcolor="rgba(34,102,170,0.2)"))
    fig.add_trace(go.Scatter(x=bins,y=h2,name="Enhanced",fill="tozeroy",
        line=dict(color="#00ccaa",width=1.5),fillcolor="rgba(0,204,170,0.15)"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3d6080",family="DM Mono"),height=200,
        margin=dict(l=20,r=10,t=10,b=30),
        legend=dict(orientation="h",yanchor="bottom",y=1.0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Pixel Intensity",gridcolor="#0d1a28",tickfont=dict(size=9)),
        yaxis=dict(title="Count",gridcolor="#0d1a28",tickfont=dict(size=9)))
    return fig

def call_claude(api_key, gray, sharpened, edges, segmented, params):
    q = params.get("_quality",{})
    pstr = (f"Blur={params['blur_k']}, CLAHE={params['clahe_clip']}, "
            f"Canny={params['canny_low']}–{params['canny_high']}, "
            f"Sharpen={params['sharpen_str']}, Seg={params['seg_method']}, Morph={params['morph_op']}")
    sstr = (f"mean={q.get('mean',0):.1f}, std={q.get('std',0):.1f}, "
            f"sharpness={q.get('sharpness',0):.1f}, entropy={q.get('entropy',0):.2f}bits, "
            f"noise={params.get('_noise',0)}, edge_density={params.get('_ed',0)}%")
    prompt = f"""You are an expert medical imaging analyst. You receive 4 versions of the same image:
1. Original grayscale  2. CLAHE-enhanced  3. Canny edge map  4. Segmentation mask

Parameters: {pstr}
Stats: {sstr}

Write a structured report with these exact bold headings:

**IMAGE TYPE** – modality and body region
**QUALITY ASSESSMENT** – brightness, contrast, noise, sharpness; rate Excellent/Good/Fair/Poor
**ANATOMICAL STRUCTURES** – visible structures or tissue types
**ENHANCEMENT EFFECTIVENESS** – what the processing revealed
**KEY OBSERVATIONS** – asymmetries, density variations, artifacts, notable regions
**RECOMMENDATIONS** – further processing or clinical follow-up

Be concise, professional, use bullet points under each heading.
End with a one-line disclaimer that this is not a clinical diagnosis."""

    resp = requests.post("https://api.anthropic.com/v1/messages",
        headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
        json={"model":"claude-sonnet-4-20250514","max_tokens":1200,"messages":[{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":to_b64(gray)}},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":to_b64(sharpened)}},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":to_b64(edges)}},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":to_b64(segmented)}},
        ]}]},timeout=90)
    if resp.status_code!=200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")
    return resp.json()["content"][0]["text"]


# ─── HERO ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🔬 MedVision AI</div>
  <div class="hero-sub">Medical Image Enhancement & AI Analysis System</div>
  <div class="hero-badges">
    <span class="badge">Auto-Optimize</span>
    <span class="badge">Claude Vision AI</span>
    <span class="badge green">CLAHE · Canny · Otsu · Morphology</span>
    <span class="badge amber">Research Use Only</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── UPLOAD ──────────────────────────────────────────────────
uploaded_file = st.file_uploader("Drop medical image here",
    type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""<div style="text-align:center;padding:3rem;color:#1e3a5a;
    font-family:'DM Mono',monospace;font-size:0.85rem;letter-spacing:0.1em;">
    ↑ UPLOAD A MEDICAL IMAGE TO BEGIN<br>
    <span style="font-size:0.7rem;color:#152838;">JPG · PNG · JPEG  ·  X-ray · MRI · CT · Histology</span>
    </div>""", unsafe_allow_html=True)
    st.stop()

try:
    pil_img, bgr, gray = load_image(uploaded_file)
except Exception as e:
    st.error(f"❌ Could not load image: {e}")
    st.stop()

auto_p = auto_tune(gray)
quality = auto_p["_quality"]

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Sora,sans-serif;font-weight:700;font-size:1rem;color:#5bb8ff;padding:0.5rem 0 1rem 0;">⚙️ Pipeline Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="auto-info">✦ Auto-optimize active<br>All sliders pre-set from image stats.<br>Adjust freely to fine-tune.</div>', unsafe_allow_html=True)
    auto_mode = st.toggle("🔄 Auto-Optimize", value=True)

    def ap(key, default): return auto_p[key] if auto_mode else default

    st.markdown('<div class="sb-section">Noise Reduction</div>', unsafe_allow_html=True)
    blur_k   = st.slider("Gaussian Blur Kernel",1,15,ap("blur_k",5),step=2)
    median_k = st.slider("Median Filter",1,15,ap("median_k",3),step=2)

    st.markdown('<div class="sb-section">Contrast Enhancement</div>', unsafe_allow_html=True)
    clahe_clip  = st.slider("CLAHE Clip Limit",1.0,8.0,float(ap("clahe_clip",2.0)),step=0.5)
    clahe_tile  = st.slider("CLAHE Tile Grid",4,16,ap("clahe_tile",8),step=4)
    sharpen_str = st.slider("Sharpening Strength",0.0,3.0,float(ap("sharpen_str",1.0)),step=0.1)

    st.markdown('<div class="sb-section">Edge Detection</div>', unsafe_allow_html=True)
    canny_low  = st.slider("Canny Low", 0,255,ap("canny_low",50))
    canny_high = st.slider("Canny High",0,255,ap("canny_high",150))

    st.markdown('<div class="sb-section">Segmentation</div>', unsafe_allow_html=True)
    seg_opts   = ["Binary","Otsu","Adaptive Mean","Adaptive Gaussian"]
    seg_method = st.selectbox("Method",seg_opts,index=seg_opts.index(ap("seg_method","Otsu")))
    thresh_val = st.slider("Binary Threshold",0,255,ap("thresh_val",128),
                           disabled=seg_method!="Binary")

    st.markdown('<div class="sb-section">Morphology</div>', unsafe_allow_html=True)
    morph_opts = ["None","Dilate","Erode","Open","Close"]
    morph_op = st.selectbox("Operation",morph_opts,index=morph_opts.index(ap("morph_op","None")))
    morph_k  = st.slider("Morph Kernel",3,15,ap("morph_k",5),step=2)

    st.divider()
    st.markdown('<div class="sb-section">Claude Vision AI</div>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key",type="password",placeholder="sk-ant-...")
    st.caption("🔗 [Get free API key →](https://console.anthropic.com)")
    run_ai  = st.button("🧠 Run AI Analysis",use_container_width=True,
                        disabled=not bool(api_key),type="primary")

# ─── PIPELINE ────────────────────────────────────────────────
params = {"blur_k":blur_k,"median_k":median_k,"clahe_clip":clahe_clip,"clahe_tile":clahe_tile,
          "sharpen_str":sharpen_str,"canny_low":canny_low,"canny_high":canny_high,
          "seg_method":seg_method,"thresh_val":thresh_val,"morph_op":morph_op,"morph_k":morph_k,
          "_quality":quality,"_noise":auto_p["_noise"],"_ed":auto_p["_ed"]}

denoised, enhanced, sharpened, edges, segmented = run_pipeline(gray, params)
post_q = image_quality(sharpened)

# ─── METRICS ─────────────────────────────────────────────────
qr = ("Excellent" if quality["sharpness"]>500 else
      "Good"      if quality["sharpness"]>100 else
      "Fair"      if quality["sharpness"]>20  else "Poor")
ed = auto_p["_ed"]

st.markdown(f"""
<div class="metrics-row">
  <div class="metric-card">
    <div class="metric-label">Mean Intensity</div>
    <div class="metric-value">{quality['mean']:.0f}<span class="metric-unit">/255</span></div>
    <div class="metric-sub">↳ {quality['min']} – {quality['max']} range</div>
  </div>
  <div class="metric-card {'good' if quality['std']>40 else 'warn'}">
    <div class="metric-label">Contrast (Std)</div>
    <div class="metric-value">{quality['std']:.1f}</div>
    <div class="metric-sub">{'✓ Good contrast' if quality['std']>40 else '⚠ Low contrast'}</div>
  </div>
  <div class="metric-card {'good' if quality['sharpness']>100 else 'warn'}">
    <div class="metric-label">Sharpness</div>
    <div class="metric-value">{quality['sharpness']:.0f}</div>
    <div class="metric-sub">{qr}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Edge Density</div>
    <div class="metric-value">{ed:.2f}<span class="metric-unit">%</span></div>
    <div class="metric-sub">Structural complexity</div>
  </div>
  <div class="metric-card {'good' if quality['entropy']>6 else 'warn'}">
    <div class="metric-label">Entropy</div>
    <div class="metric-value">{quality['entropy']:.2f}<span class="metric-unit">bits</span></div>
    <div class="metric-sub">Information content</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── ORIGINAL vs ENHANCED ────────────────────────────────────
st.markdown('<div class="section-title">📷 Original vs Enhanced</div>', unsafe_allow_html=True)
c1,c2 = st.columns(2)
c1.image(gray,      caption="Original Grayscale",use_container_width=True)
c2.image(sharpened, caption=f"Enhanced · CLAHE {clahe_clip} · Sharpen {sharpen_str}",use_container_width=True)

# ─── HISTOGRAM ───────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Intensity Distribution</div>', unsafe_allow_html=True)
st.plotly_chart(histogram_fig(gray, sharpened), use_container_width=True)

# ─── PIPELINE STAGES ─────────────────────────────────────────
st.markdown('<div class="section-title">🔬 Processing Pipeline</div>', unsafe_allow_html=True)
p1,p2,p3 = st.columns(3)
p1.image(denoised,  caption=f"① Denoised  (Gaussian k={blur_k} + Median k={median_k})",use_container_width=True)
p2.image(edges,     caption=f"② Edge Map  (Canny {canny_low}–{canny_high})",use_container_width=True)
p3.image(segmented, caption=f"③ Segmented  ({seg_method} + Morph {morph_op})",use_container_width=True)

# ─── AUTO-OPTIMIZE DIAGNOSTICS ───────────────────────────────
with st.expander("🔍 Auto-Optimize Parameter Reasoning", expanded=False):
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{r}</td></tr>"
                   for k,(v,r) in auto_p["_reasons"].items())
    st.markdown(f"""<table class="diag-table">
      <thead><tr><th>Parameter</th><th>Auto Value</th><th>Reasoning</th></tr></thead>
      <tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

# ─── AI INSIGHTS ─────────────────────────────────────────────
st.markdown('<div class="section-title">🧠 Claude AI Clinical Analysis</div>', unsafe_allow_html=True)

if not api_key:
    st.markdown("""<div style="background:#080f1a;border:1px dashed #1a3050;border-radius:10px;
    padding:1.5rem 2rem;color:#1e3a5a;font-family:'DM Mono',monospace;font-size:0.82rem;
    line-height:1.8;text-align:center;">
    Enter your <span style="color:#2a5a8a">Anthropic API key</span> in the sidebar<br>
    and click <span style="color:#2a5a8a">🧠 Run AI Analysis</span> to get<br>
    a structured clinical-style report from Claude Vision.<br><br>
    <span style="font-size:0.7rem;color:#152838;">Free API key → console.anthropic.com</span>
    </div>""", unsafe_allow_html=True)
elif run_ai:
    with st.spinner("Claude Vision is examining your images…"):
        try:
            insights = call_claude(api_key, gray, sharpened, edges, segmented, params)
            st.session_state["ai_insights"] = insights
        except Exception as e:
            st.error(f"AI analysis failed: {e}")

if "ai_insights" in st.session_state:
    st.markdown('<div class="ai-insights"><div class="ai-header">⚕ Claude Vision · Clinical Analysis Report</div></div>', unsafe_allow_html=True)
    st.markdown(st.session_state["ai_insights"])
    st.markdown('<div class="ai-disclaimer">⚠ Automated AI analysis for research/educational use only. NOT a clinical diagnosis. Consult a qualified radiologist for medical decisions.</div>', unsafe_allow_html=True)

# ─── DOWNLOADS ───────────────────────────────────────────────
st.markdown('<div class="section-title">⬇️ Export Results</div>', unsafe_allow_html=True)
outputs = {"01_original_gray":gray,"02_denoised":denoised,"03_enhanced":enhanced,
           "04_sharpened":sharpened,"05_edges":edges,"06_segmented":segmented}
dcols = st.columns(len(outputs)+1)
for col,(name,arr) in zip(dcols,outputs.items()):
    col.download_button(name.split("_",1)[1].replace("_"," ").title(),
        to_png(arr),f"{name}.png","image/png",use_container_width=True)
dcols[-1].download_button("⬇️ All ZIP",build_zip(outputs),
    "medvision_results.zip","application/zip",use_container_width=True)

st.markdown("""<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
border-top:1px solid #0d1825;color:#152838;font-family:'DM Mono',monospace;
font-size:0.68rem;letter-spacing:0.1em;">
MedVision AI · Research & Educational Use Only · Not a Medical Device
</div>""", unsafe_allow_html=True)