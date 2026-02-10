import numpy as np

# --- NLP Explanation Module ---
class DiabeticRetinopathyExplainer:
    def __init__(self, threshold=0.7):
        # 1. Predefined Explanations with Keywords:
        self.explanation_templates = {
            "microaneurysms": [
                "The model found evidence of microaneurysms, which are small, localized bulges in the blood vessels of the retina.",
                "The highlighted regions suggest the presence of microaneurysms, indicating early signs of diabetic retinopathy.",
            ],
            "exudates": [
                "The model detected exudates, which are yellow deposits of fats and proteins that leak from damaged blood vessels.",
                "The activation in these areas might be due to exudates, a common finding in diabetic retinopathy.",
            ],
            "hemorrhages": [
                "The model identified potential hemorrhages, which are areas of bleeding within the retina.",
                "The highlighted regions could indicate hemorrhages, suggesting a more advanced stage of diabetic retinopathy.",
            ],
            "neovascularization": [
                "The model found signs of neovascularization, which is the formation of new, abnormal blood vessels.",
                "The activation pattern suggests neovascularization, a serious complication of diabetic retinopathy that can lead to vision loss.",
            ],
            "normal": [
                "The model did not find strong evidence of diabetic retinopathy in the highlighted regions.",
                "The image appears relatively normal based on the model's analysis.",
            ],
        }

        # 2. Keyword Mapping to Regions (Simplified for this Example):
        self.keyword_regions = {
            "microaneurysms": [(0.2, 0.8), (0.2, 0.8)],  # Central region
            "exudates": [(0.1, 0.9), (0.1, 0.9)],  # Most of the image
            "hemorrhages": [(0.3, 0.7), (0.3, 0.7)],  # Central-ish region
            "neovascularization": [(0.0, 1.0), (0.0, 1.0)],  # Anywhere
        }

        self.threshold = threshold

    def analyze_heatmap_regions(self, heatmap):
        """
        Analyzes the heatmap to find the most activated regions.

        Args:
            heatmap: The Grad-CAM heatmap.

        Returns:
            A list of keywords corresponding to the activated regions.
        """
        activated_regions = []
        normalized_heatmap = heatmap / heatmap.max()

        for keyword, [(x_min, x_max), (y_min, y_max)] in self.keyword_regions.items():
            region_activation = normalized_heatmap[
                int(y_min * heatmap.shape[0]) : int(y_max * heatmap.shape[0]),
                int(x_min * heatmap.shape[1]) : int(x_max * heatmap.shape[1]),
            ]

            if region_activation.mean() > self.threshold:
                activated_regions.append(keyword)

        return activated_regions

    def generate_explanation(self, activated_regions):
        """
        Generates a text explanation based on activated regions.

        Args:
            activated_regions: A list of keywords corresponding to activated regions.

        Returns:
            A text explanation.
        """
        if not activated_regions:
            return self.explanation_templates["normal"][0]

        explanation = ""
        for region in activated_regions:
            if region in self.explanation_templates:
                explanation += (
                    np.random.choice(self.explanation_templates[region]) + " "
                )

        return explanation.strip()
