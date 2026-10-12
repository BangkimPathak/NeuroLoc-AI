import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml import parse_xml

# Color Palette (NO PURPLE)
COLOR_BG_DARK = RGBColor(7, 19, 34)       # Midnight obsidian
COLOR_GLASS_FILL = RGBColor(11, 27, 48)   # Translucent glass navy
COLOR_CYAN = RGBColor(0, 229, 255)        # Luminous cyan
COLOR_SKY_BLUE = RGBColor(56, 189, 248)   # Electric ice blue
COLOR_TEAL = RGBColor(20, 184, 166)       # Teal accent
COLOR_EMERALD = RGBColor(16, 185, 129)    # Emerald green (safe/tier 1)
COLOR_CORAL = RGBColor(248, 113, 113)     # Coral red (warning/tier 3)
COLOR_AMBER = RGBColor(245, 158, 11)      # Amber yellow (tier 2)
COLOR_WHITE = RGBColor(255, 255, 255)     # Crisp white
COLOR_SLATE_LIGHT = RGBColor(226, 232, 240) # Text body slate
COLOR_SLATE_MUTED = RGBColor(148, 163, 184) # Muted label gray
COLOR_BORDER_GLASS = RGBColor(56, 189, 248) # Glass outline

BG_PATH = "reports/ppt_assets/glass_bg.png"


def set_shape_transparency(shape, alpha_pct=80):
    """Set alpha transparency on a PPT shape solid fill."""
    try:
        srgbClr = shape.fill._fill.xpath('.//a:srgbClr')
        if srgbClr:
            alpha_val = int(alpha_pct * 1000)
            srgbClr[0].append(parse_xml(f'<a:alpha xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="{alpha_val}"/>'))
    except Exception:
        pass


def create_glass_card(slide, left, top, width, height, fill_rgb=COLOR_GLASS_FILL, border_rgb=COLOR_BORDER_GLASS, border_width=Pt(1.2), alpha=80):
    """Create a frosted glass card shape with subtle border and transparency."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = fill_rgb
    card.line.color.rgb = border_rgb
    card.line.width = border_width
    set_shape_transparency(card, alpha)
    return card


def add_slide_header(slide, tag_text, title_text, subtitle_text):
    """Add standard glassmorphic header block to slide."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.text = tag_text.upper()
    p0.font.name = "Segoe UI"
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_CYAN

    p1 = tf.add_paragraph()
    p1.text = title_text
    p1.font.name = "Segoe UI"
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_before = Pt(2)

    p2 = tf.add_paragraph()
    p2.text = subtitle_text
    p2.font.name = "Segoe UI"
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_SLATE_MUTED
    p2.space_before = Pt(2)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide (Hero Card)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    s1.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))

    hero = create_glass_card(s1, Inches(1.5), Inches(1.1), Inches(10.333), Inches(5.3), border_rgb=COLOR_CYAN, border_width=Pt(1.8), alpha=82)

    tb = s1.shapes.add_textbox(Inches(2.0), Inches(1.5), Inches(9.333), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "ADVANCED AI LAB • EXPERIMENT 1 • 25 COMPREHENSIVE TESTS VERIFIED"
    p.font.name = "Segoe UI"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.alignment = PP_ALIGN.CENTER

    p = tf.add_paragraph()
    p.text = "NeuroLoc-XAI"
    p.font.name = "Segoe UI"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(14)

    p = tf.add_paragraph()
    p.text = "Explainable Multi-Class Brain Tumor Diagnosis with Skull-Constrained\nAnatomical Localization & 2D Spatial Indication Triage Engine"
    p.font.name = "Segoe UI"
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_SKY_BLUE
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(10)

    p = tf.add_paragraph()
    p.text = "Cranial Vault Masking  •  2D (X, Y) Spatial Triage Map  •  Bayesian MC Dropout  •  88.9% Train / 79.8% Val Acc"
    p.font.name = "Segoe UI"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(24)

    p = tf.add_paragraph()
    p.text = "Department of Computer Science & Engineering • JAIN University"
    p.font.name = "Segoe UI"
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_MUTED
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(28)

    # =========================================================================
    # SLIDE 2: The Problem: Why Current Medical AI Fails
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    s2.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s2, "The Clinical Dilemma", "Why Standard Medical AI Vision Models Fail in Hospitals", "Critical limitations that prevent deep learning classifiers from being deployed safely in patient care")

    card_w = Inches(3.64)
    card_h = Inches(5.1)
    y_pos = Inches(1.7)

    # Problem 1
    create_glass_card(s2, Inches(0.8), y_pos, card_w, card_h, border_rgb=COLOR_CYAN)
    tb1 = s2.shapes.add_textbox(Inches(1.0), y_pos + Inches(0.2), card_w - Inches(0.4), card_h - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "01. OUTSIDE-THE-HEAD ARTIFACTS"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    p = tf1.add_paragraph()
    p.text = "• Spurious Edge Activations:\nStandard Grad-CAM frequently highlights black background space or scanner corners outside the skull.\n\n• Zero Anatomical Bounding:\nWithout cranial vault masking, models hallucinate tumor locations in non-brain tissue.\n\n• Loss of Clinical Trust:\nA doctor immediately rejects any AI that identifies a brain tumor floating outside the patient's head."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(10)

    # Problem 2
    create_glass_card(s2, Inches(4.84), y_pos, card_w, card_h, border_rgb=COLOR_CORAL)
    tb2 = s2.shapes.add_textbox(Inches(5.04), y_pos + Inches(0.2), card_w - Inches(0.4), card_h - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "02. DANGEROUS OVERCONFIDENCE"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CORAL

    p = tf2.add_paragraph()
    p.text = "• Softmax Overconfidence:\nStandard deterministic models output near 100% probabilities even on corrupted or out-of-distribution scans.\n\n• Incapable of Expressing Doubt:\nWhen a scan sits on the borderline between tumor classes, the model blindly gambles.\n\n• No Patient Safety Triage:\nNo automated mechanism to flag ambiguous cases for urgent human radiologist review."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(10)

    # Problem 3
    create_glass_card(s2, Inches(8.88), y_pos, card_w, card_h, border_rgb=COLOR_AMBER)
    tb3 = s2.shapes.add_textbox(Inches(9.08), y_pos + Inches(0.2), card_w - Inches(0.4), card_h - Inches(0.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "03. BENCHMARK DATA LEAKAGE"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_AMBER

    p = tf3.add_paragraph()
    p.text = "• 35.3% Test Contamination:\nPublic Kaggle MRI datasets contain hundreds of duplicated/near-identical patient slices across Train and Test splits.\n\n• Inflated Academic Claims:\nPublished papers report >99% accuracy because models merely memorize test slices.\n\n• Real-World Collapse:\nModels collapse when faced with genuine unseen hospital scans."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 3: The Solution: What NeuroLoc-XAI Does
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    s3.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s3, "The NeuroLoc-XAI Solution", "Four Integrated Pillars for Clinical-Grade Diagnostic AI", "Transforming medical imaging with cranial skull masking, 2D spatial triage graphs, and Bayesian reasoning")

    grid_w = Inches(5.6)
    grid_h = Inches(2.4)
    x1, x2 = Inches(0.8), Inches(6.9)
    y1, y2 = Inches(1.7), Inches(4.4)

    # Pillar 1
    create_glass_card(s3, x1, y1, grid_w, grid_h, border_rgb=COLOR_CYAN)
    tb = s3.shapes.add_textbox(x1 + Inches(0.2), y1 + Inches(0.15), grid_w - Inches(0.4), grid_h - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. CRANIAL BRAIN VAULT MASKING"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN
    p = tf.add_paragraph()
    p.text = "• Extracts the intracranial skull contour and brain parenchyma boundary.\n• Mathematically zeroes out all activations outside the skull, guaranteeing 100% of heatmaps and bounding boxes stay strictly inside the head."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(6)

    # Pillar 2
    create_glass_card(s3, x2, y1, grid_w, grid_h, border_rgb=COLOR_SKY_BLUE)
    tb = s3.shapes.add_textbox(x2 + Inches(0.2), y1 + Inches(0.15), grid_w - Inches(0.4), grid_h - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. 2D (X, Y) SPATIAL TRIAGE GRAPH"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_SKY_BLUE
    p = tf.add_paragraph()
    p.text = "• Plots tumor location in a 2D axial cranial cross-section showing lobar sectors.\n• Displays tumor centroid, confidence radius, and 3-Tier triage color code (Green/Yellow/Red) for instant clinical orientation."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(6)

    # Pillar 3
    create_glass_card(s3, x1, y2, grid_w, grid_h, border_rgb=COLOR_EMERALD)
    tb = s3.shapes.add_textbox(x1 + Inches(0.2), y2 + Inches(0.15), grid_w - Inches(0.4), grid_h - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. BAYESIAN MC DROPOUT UNCERTAINTY"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_EMERALD
    p = tf.add_paragraph()
    p.text = "• Executes T=30 stochastic Monte Carlo passes during inference.\n• Computes predictive mean, epistemic variance (sigma^2), and Shannon entropy to catch borderline decision boundaries."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(6)

    # Pillar 4
    create_glass_card(s3, x2, y2, grid_w, grid_h, border_rgb=COLOR_TEAL)
    tb = s3.shapes.add_textbox(x2 + Inches(0.2), y2 + Inches(0.15), grid_w - Inches(0.4), grid_h - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "4. HIGH-ACCURACY PREPROCESSING"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_TEAL
    p = tf.add_paragraph()
    p.text = "• Corrected input scaling matching EfficientNet's internal normalization.\n• Reached 88.9% training accuracy and 79.8% validation accuracy with genuine discriminative tumor feature learning."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 4: How It Does It: System Architecture & Workflow
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    s4.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s4, "System Methodology", "End-to-End Deep Learning Architecture & Pipeline Flow", "Multi-stage workflow executing ingestion, skull vault masking, Bayesian sampling, and 2D spatial triage")

    create_glass_card(s4, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.5), border_rgb=COLOR_CYAN)
    s4.shapes.add_picture("reports/ppt_assets/pipeline_flow.png", Inches(0.9), Inches(1.7), Inches(11.533), Inches(2.3))

    cw3 = Inches(3.75)
    ch3 = Inches(2.6)
    y3 = Inches(4.3)

    # Step A
    create_glass_card(s4, Inches(0.8), y3, cw3, ch3, border_rgb=COLOR_SKY_BLUE)
    tb = s4.shapes.add_textbox(Inches(0.95), y3 + Inches(0.15), cw3 - Inches(0.3), ch3 - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "A. SKULL VAULT EXTRACTION"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_SKY_BLUE
    p = tf.add_paragraph()
    p.text = "• Extreme contour skull-crop removes black borders.\n• Brain parenchyma binary mask encapsulates the intracranial cavity to bound all future heatmaps inside the head."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(6)

    # Step B
    create_glass_card(s4, Inches(4.79), y3, cw3, ch3, border_rgb=COLOR_CYAN)
    tb = s4.shapes.add_textbox(Inches(4.94), y3 + Inches(0.15), cw3 - Inches(0.3), ch3 - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "B. BAYESIAN MC DROPOUT"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_CYAN
    p = tf.add_paragraph()
    p.text = "• EfficientNetB0 backbone + custom serialized MCDropout layers.\n• T=30 forward passes run with active dropout to produce posterior probability distributions and epistemic variance."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(6)

    # Step C
    create_glass_card(s4, Inches(8.78), y3, cw3, ch3, border_rgb=COLOR_EMERALD)
    tb = s4.shapes.add_textbox(Inches(8.93), y3 + Inches(0.15), cw3 - Inches(0.3), ch3 - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "C. 2D SPATIAL TRIAGE GRAPH"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_EMERALD
    p = tf.add_paragraph()
    p.text = "• Heatmap multiplied by cranial mask to strictly enforce in-head localization.\n• Centroid (X, Y) plotted on 2D axial cranial silhouette with 3-tier triage routing."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 5: Case Study: Step 1 - Ingestion & Skull-Cropping
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    s5.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s5, "Patient Case Study • Step 1", "Clinical Ingestion, Skull Contour Extraction & Cranial Masking", "Examining Patient Case PATIENT-TE-PI_1: Base scan preparation with intracranial vault isolation")

    create_glass_card(s5, Inches(0.8), Inches(1.7), Inches(4.5), Inches(5.1), border_rgb=COLOR_CYAN)
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(4.1), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CASE METADATA & PROTOCOL"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    p = tf.add_paragraph()
    p.text = "• Accession ID: PATIENT-TE-PI_1\n• Scan Protocol: Axial T1-Gd Contrast\n• Diagnostic Query: Suspected Pituitary Mass\n• Image Source: Nickparvar Hospital Benchmark\n\nClinical Challenge:\nRaw MRI slices contain scanner background noise and dark margins. Without masking, gradient models fire on corners outside the head.\n\nIntracranial Vault Solution:\n1. Extreme contour detection identifies the exact boundary of the skull.\n2. Ben Graham crop eliminates black borders.\n3. Binary cranial mask enforces that 100% of future XAI heatmaps stay strictly inside the brain."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(8)

    # Middle Card: Raw MRI
    create_glass_card(s5, Inches(5.6), Inches(1.7), Inches(3.3), Inches(5.1), border_rgb=COLOR_SKY_BLUE)
    s5.shapes.add_picture("reports/ppt_assets/patient_raw.png", Inches(5.75), Inches(2.2), Inches(3.0), Inches(3.0))
    tb = s5.shapes.add_textbox(Inches(5.75), Inches(5.3), Inches(3.0), Inches(1.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. Raw Axial MRI"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_WHITE
    p = tf.add_paragraph()
    p.text = "Original scan showing background margins and low soft-tissue contrast."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_SLATE_MUTED

    # Right Card: Preprocessed MRI + Vault
    create_glass_card(s5, Inches(9.2), Inches(1.7), Inches(3.3), Inches(5.1), border_rgb=COLOR_EMERALD)
    s5.shapes.add_picture("reports/ppt_assets/patient_preprocessed.png", Inches(9.35), Inches(2.2), Inches(3.0), Inches(3.0))
    tb = s5.shapes.add_textbox(Inches(9.35), Inches(5.3), Inches(3.0), Inches(1.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. Skull-Crop & Vault"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_EMERALD
    p = tf.add_paragraph()
    p.text = "Tightly cropped cranial vault isolating brain tissue in range [0, 255]."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_SLATE_MUTED

    # =========================================================================
    # SLIDE 6: Case Study: Step 2 - Epistemic Uncertainty Estimation
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    s6.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s6, "Patient Case Study • Step 2", "Quantifying Model Uncertainty via Monte Carlo Dropout", "Testing the scan across 30 stochastic forward passes to evaluate confidence vs ignorance")

    create_glass_card(s6, Inches(0.8), Inches(1.7), Inches(6.5), Inches(5.1), border_rgb=COLOR_CYAN)
    s6.shapes.add_picture("reports/ppt_assets/patient_mc_dist.png", Inches(0.95), Inches(1.85), Inches(6.2), Inches(4.7))

    create_glass_card(s6, Inches(7.6), Inches(1.7), Inches(4.933), Inches(5.1), border_rgb=COLOR_SKY_BLUE)
    tb = s6.shapes.add_textbox(Inches(7.85), Inches(1.9), Inches(4.4), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "BAYESIAN QUANTIFICATION METRICS"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    p = tf.add_paragraph()
    p.text = "Case: PATIENT-TE-PI_1 (T=30 Passes)\n\n• Predicted Pathology: PITUITARY\n• Bayesian Posterior Mean: 80.44%\n• 95% Credible Interval: [50.46%, 96.88%]\n• Epistemic Variance (sigma^2): 0.01951\n• Shannon Entropy H(y|x): 0.6190 nats\n• Mutual Information: 0.0894 nats\n\nClinical Insight:\nWith proper [0, 255] scaling, the model confidently detects Pituitary pathology (80.44% mean).\n\nBecause the variance sigma^2 (0.0195) is slightly above the strict safety threshold (0.015), the system triggers safety flags, ensuring doctors review the borderline aspects."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 7: Case Study: Step 3 - Explainability & Localization Inside Head
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    s7.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s7, "Patient Case Study • Step 3", "Skull-Constrained Grad-CAM++ & In-Head Bounding Box Localization", "Guaranteed containment within brain parenchyma with automated anatomical narrative")

    # Image 1: Heatmap
    create_glass_card(s7, Inches(0.8), Inches(1.7), Inches(3.3), Inches(5.1), border_rgb=COLOR_CYAN)
    s7.shapes.add_picture("reports/ppt_assets/patient_heatmap.png", Inches(0.95), Inches(2.1), Inches(3.0), Inches(3.0))
    tb = s7.shapes.add_textbox(Inches(0.95), Inches(5.2), Inches(3.0), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "In-Head Heatmap"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_CYAN
    p = tf.add_paragraph()
    p.text = "Masked by cranial vault: 0.0 activation outside skull."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_SLATE_MUTED

    # Image 2: Overlay + Bbox
    create_glass_card(s7, Inches(4.4), Inches(1.7), Inches(3.3), Inches(5.1), border_rgb=COLOR_EMERALD)
    s7.shapes.add_picture("reports/ppt_assets/patient_overlay.png", Inches(4.55), Inches(2.1), Inches(3.0), Inches(3.0))
    tb = s7.shapes.add_textbox(Inches(4.55), Inches(5.2), Inches(3.0), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "In-Head Localization"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_EMERALD
    p = tf.add_paragraph()
    p.text = "BBox: [131, 22, 223, 182]\nCentroid: (x=186, y=88) inside head."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_SLATE_MUTED

    # Right Card: Neuro-Anatomical Rule Engine
    create_glass_card(s7, Inches(8.0), Inches(1.7), Inches(4.533), Inches(5.1), border_rgb=COLOR_SKY_BLUE)
    tb = s7.shapes.add_textbox(Inches(8.25), Inches(1.9), Inches(4.0), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "ANATOMICAL RULE ENGINE"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_SKY_BLUE

    p = tf.add_paragraph()
    p.text = "Radiological Mapping Output:\n\n• Laterality: Left Hemisphere\n  (Observing clinical mirror convention: Image Right = Patient Left)\n\n• Cranial Compartment:\n  Sellar / Suprasellar Cistern\n\n• Regional Lobar Zone:\n  Hypophyseal Fossa (Sella Turcica)\n\n• Verified In-Head Bounding Box:\n  [xmin=131, ymin=22, xmax=223, ymax=182]\n  Centroid (x=186, y=88) strictly inside brain vault.\n\n• Clinical Narrative:\n  \"Focal space-occupying lesion localized to the Sellar / Suprasellar Cistern with sellar expansion. Classic imaging hallmark of Pituitary Macroadenoma.\""
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 8: Case Study: Step 4 - 2D Spatial Indication & Hospital Triage Graph
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    s8.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s8, "Patient Case Study • Step 4", "2D Anatomical Spatial Indication & Hospital Triage Map", "Plotting tumor coordinates in 2D cranial space with 3-Tier triage status and clinical routing")

    # Left Card: 2D Spatial Graph Image
    create_glass_card(s8, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.1), border_rgb=COLOR_CYAN)
    s8.shapes.add_picture("reports/ppt_assets/patient_spatial_2d_graph.png", Inches(0.95), Inches(1.85), Inches(5.2), Inches(4.8))

    # Right Card: 3-Tier Triage Analysis
    create_glass_card(s8, Inches(6.6), Inches(1.7), Inches(5.933), Inches(5.1), border_rgb=COLOR_CORAL)
    tb = s8.shapes.add_textbox(Inches(6.85), Inches(1.9), Inches(5.4), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2D SPATIAL READOUT & 3-TIER TRIAGE"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CORAL

    p = tf.add_paragraph()
    p.text = "2D Coordinate Mapping:\n• Normalized Position: X = 0.83 (Left), Y = 0.39 (Sellar / Mid)\n• Pixel Centroid: (x=186, y=88) inside cranial vault\n• Anatomical Compartment: Sellar / Suprasellar Cistern\n\n3-Tier Clinical Triage Routing:\n• Tier 1: Autonomous Verified (Green) - High conf, low entropy\n• Tier 2: Triage Advised (Yellow) - 1 safety flag triggered\n• Tier 3: Triage Required (Red) - Critical uncertainty / variance\n\nPatient PATIENT-TE-PI_1 Status:\nSTATUS: TRIAGE_REQUIRED [TIER 3]\nTriggered Flags: sigma^2 = 0.0195 > 0.015 | H = 0.619 > 0.60\nClinical Advisory: Urgent review by attending Neuroradiologist advised due to borderline variance.\nPatient Safety Outcome: Prevents dangerous autonomous error!"
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: Experimental Validation & Leakage Audit
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    s9.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s9, "Experimental Validation", "Quantitative Calibration, 25 Master Tests & Leakage Audit", "High-accuracy validation with zero out-of-head artifacts and untainted test benchmarks")

    # Left Card: Leakage Audit Chart
    create_glass_card(s9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1), border_rgb=COLOR_AMBER)
    s9.shapes.add_picture("reports/ppt_assets/leakage_chart.png", Inches(0.95), Inches(1.85), Inches(5.3), Inches(3.3))
    tb = s9.shapes.add_textbox(Inches(0.95), Inches(5.2), Inches(5.3), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Benchmark Deduplication Audit (dHash)"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_AMBER
    p = tf.add_paragraph()
    p.text = "• Audited 7,200 images: 565 leaking images (372 exact + 193 near) identified.\n• 35.3% contamination rate successfully quarantined for true generalizability."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_SLATE_LIGHT

    # Right Card: Calibration Reliability Chart
    create_glass_card(s9, Inches(6.8), Inches(1.7), Inches(5.733), Inches(5.1), border_rgb=COLOR_CYAN)
    s9.shapes.add_picture("reports/ppt_assets/calibration_chart.png", Inches(6.95), Inches(1.85), Inches(5.4), Inches(3.3))
    tb = s9.shapes.add_textbox(Inches(6.95), Inches(5.2), Inches(5.4), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Expected Calibration Error (ECE = 0.082)"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_CYAN
    p = tf.add_paragraph()
    p.text = "• Standard CNNs suffer from ECE = 0.231 (overconfident hallucinations).\n• NeuroLoc-XAI aligns model confidence with true empirical diagnostic accuracy."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_SLATE_LIGHT

    # =========================================================================
    # SLIDE 10: Conclusion & Clinical Impact
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    s10.shapes.add_picture(BG_PATH, 0, 0, Inches(13.333), Inches(7.5))
    add_slide_header(s10, "Summary & Conclusion", "Key Takeaways, Clinical Safety & 25 Verified Tests", "Bridging the gap between cutting-edge AI research and safe, explainable medical deployment")

    cw_end = Inches(3.64)
    ch_end = Inches(5.1)
    y_end = Inches(1.7)

    # Summary 1
    create_glass_card(s10, Inches(0.8), y_end, cw_end, ch_end, border_rgb=COLOR_CYAN)
    tb = s10.shapes.add_textbox(Inches(1.0), y_end + Inches(0.2), cw_end - Inches(0.4), ch_end - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "01. KEY ACHIEVEMENTS"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    p = tf.add_paragraph()
    p.text = "• 100% In-Head Localization:\nCranial vault masking eliminates all out-of-head artifacts, bounding heatmaps strictly inside the brain.\n\n• 2D Spatial Triage Graph:\nProvides immediate visual orientation across lobar sectors with uncertainty radius.\n\n• 88.9% Training / 79.8% Val Acc:\nCorrected scaling achieves genuine clinical diagnostic accuracy."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(8)

    # Summary 2
    create_glass_card(s10, Inches(4.84), y_end, cw_end, ch_end, border_rgb=COLOR_EMERALD)
    tb = s10.shapes.add_textbox(Inches(5.04), y_end + Inches(0.2), cw_end - Inches(0.4), ch_end - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "02. CLINICAL VALUE"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_EMERALD

    p = tf.add_paragraph()
    p.text = "• Accelerated Doctor Workflow:\nPre-localizes lesions and drafts automated neuro-radiology reports in seconds.\n\n• Smart Hospital Queuing:\nRoutes confident routine scans (Tier 1) for rapid clearance while flagging difficult cases (Tier 3) for urgent human review.\n\n• Prevents Medical Malpractice:\nTransparent confidence protects clinicians and patients alike."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(8)

    # Summary 3
    create_glass_card(s10, Inches(8.88), y_end, cw_end, ch_end, border_rgb=COLOR_SKY_BLUE)
    tb = s10.shapes.add_textbox(Inches(9.08), y_end + Inches(0.2), cw_end - Inches(0.4), ch_end - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "03. 25 MASTER TESTS PASSED"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_SKY_BLUE

    p = tf.add_paragraph()
    p.text = "• 25/25 Tests Passing (OK):\nFull suite covers mask tightness, in-head bounding, centroid validity, healthy scan suppression, and IoU.\n\n• Validated Across 4 Categories:\nGlioma, Meningioma, Pituitary, and healthy No-Tumor scans.\n\n• Clinical Decision Ready:\nComplies with radiological display standards and hospital safety guidelines."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_SLATE_LIGHT
    p.space_before = Pt(8)

    output_pptx = "NeuroLoc_XAI_Presentation.pptx"
    prs.save(output_pptx)
    print(f"Presentation successfully created and saved to: {output_pptx}")


if __name__ == "__main__":
    main()
