try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    print("iterstrat is installed!")
except ImportError:
    print("iterstrat not installed. Checking sklearn...")
    from sklearn.model_selection import KFold
    print("sklearn is installed.")
