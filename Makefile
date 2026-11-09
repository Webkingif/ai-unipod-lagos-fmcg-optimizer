install:
	pip install -r requirements.txt

run:
	streamlit run app.py

clean:
	rm -rf data/ __pycache__/

.PHONY: install run clean
