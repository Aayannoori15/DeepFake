from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        widget=forms.ClearableFileInput(
            attrs={
                "accept": "image/*",
            }
        )
    )


class TextAnalysisForm(forms.Form):
    text = forms.CharField(
        label="Essay",
        widget=forms.Textarea(
            attrs={
                "rows": 12,
                "placeholder": "Paste or type the essay you want to analyze...",
                "class": "textarea",
            }
        ),
    )
