# Manga Translator

It's a basic version of manga translator, which is expected to be enriched in future with movable bboxes with a possibility to change translated text in them.

You can find the test.jpg in data folder.
By now the final stage (translation) doesn't work because the translation model is too large and it kills the app.
Everything else before it works well (find bboxes with pre-trained model, change model parameters for better inference, show found bboxes on the image)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
