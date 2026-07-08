import numpy as np

class NlpService:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        
        # Predefined clinical templates with professional medical terminology
        self.explanation_templates = {
            "microaneurysms": [
                "The model detected focal areas of high activation matching the signature of microaneurysms, which are localized swellings of retinal capillaries.",
                "The highlighted regions suggest the presence of microaneurysms, indicating early vascular alterations in the retina."
            ],
            "exudates": [
                "The model highlighted regions consistent with exudates, which represent lipid and protein leakage from hyperpermeable capillaries.",
                "Significant activation in these zones corresponds with hard exudates, suggesting lipid deposits in the deep retina."
            ],
            "hemorrhages": [
                "The model identified active nodes matching retinal hemorrhages, reflecting intraretinal bleeding due to capillary rupture.",
                "The heatmap activation profile suggests potential hemorrhages, indicating moderate to severe microvascular damage."
            ],
            "neovascularization": [
                "The model flagged proliferative regions exhibiting neovascularization—the pathological growth of new, fragile retinal vessels.",
                "Vibrant activation patterns suggest neovascularization, a severe complication of advanced diabetic retinopathy that poses a high risk for vision loss."
            ],
            "normal": [
                "No significant pathological hotspots were identified by the model; the retinal structure appears largely unremarkable in the target areas.",
                "The image is evaluated as normal with no significant localized activation representing diabetic retinopathy lesions."
            ]
        }
        
        # Keyword mapping to bounding boxes (x_range, y_range) normalized coordinates
        self.keyword_regions = {
            "microaneurysms": [(0.2, 0.8), (0.2, 0.8)],       # Central region
            "exudates": [(0.1, 0.9), (0.1, 0.9)],             # Scattered regions
            "hemorrhages": [(0.3, 0.7), (0.3, 0.7)],           # Central-ish clusters
            "neovascularization": [(0.0, 1.0), (0.0, 1.0)]     # General region
        }

    def explain_heatmap(self, heatmap):
        """
        Analyzes the heatmap spatial activation and generates a clinical narrative.
        """
        if heatmap is None or heatmap.size == 0:
            return np.random.choice(self.explanation_templates["normal"])
            
        h_max = heatmap.max()
        normalized_heatmap = heatmap / h_max if h_max > 0 else heatmap
        
        activated_features = []
        height, width = heatmap.shape
        
        for feature, [(x_min, x_max), (y_min, y_max)] in self.keyword_regions.items():
            region = normalized_heatmap[
                int(y_min * height) : int(y_max * height),
                int(x_min * width) : int(x_max * width)
            ]
            
            if region.size > 0 and region.mean() > self.threshold:
                activated_features.append(feature)
                
        if not activated_features:
            return np.random.choice(self.explanation_templates["normal"])
            
        sentences = []
        for feature in activated_features:
            sentences.append(np.random.choice(self.explanation_templates[feature]))
            
        return " ".join(sentences)
