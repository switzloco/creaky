from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Fetch full leaderboard
full_lb = []
token = None
page = 1
while True:
    lb = api.competition_leaderboard_view("rsna-knee-abnormality-detection", page_token=token)
    if not lb:
        break
    full_lb.extend(lb)
    # Check if we are in this batch
    for team in lb:
        if "switzer" in str(team).lower() or getattr(team, "teamName", "").lower() == "nswitzer":
            rank = len(full_lb) - len(lb) + lb.index(team) + 1
            print(f"--> FOUND! Rank: #{rank} | Team: {team.teamName} | Score: {team.score}")
            exit(0)
    token = getattr(api, "last_page_token", None)
    # limit search to top 300
    if len(full_lb) >= 400:
        break
    page += 1

print(f"Searched {len(full_lb)} teams. Lowest score in top {len(full_lb)}: {getattr(full_lb[-1], 'score', 'N/A')}")
