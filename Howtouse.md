# &#x09;				     HOW TO USE

1. ###### Clone the repository (git clone <repo-url>).
2. ###### Enter my branch (git checkout NasirApp).
3. ###### Install Git LFS once (git lfs install), then run git lfs pull so the data files download.
4. ###### Install the packages: py -m pip install -r requirements.txt (Windows) or pip install -r requirements.txt (Mac).
5. ###### From the repository folder, run streamlit run app/app.py (Windows: py -m streamlit run app\\app.py).
6. ###### The app opens in your browser at localhost:8501 may need to input your email first.
7. ###### layout goes from game search to VADER sentiment chart to the Steam reviews comparison.
8. ###### It defaults to Age of Empires II (2013), but you can search for games by either using the search bar or the dropdown menu and clicking the game you are looking for. 
9. ###### The program shows the breakdown of the VADER sentiment from positive to neutral to negative and represents them as percentages and bar charts.
10. ###### It then compares this with Steams review based on positive sentiment 
11. ###### We see the comparison if the neutral sentiment is dropped at the bottom of the page
12. ###### If you search for a game not on the list it returns a not found message.  

