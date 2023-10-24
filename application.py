import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plotly.express as px
import plotly.graph_objs as go

st.image("forge2.png", use_column_width=True)

# Load your data here
try:
    df_teams = pd.read_csv('region-team.csv')
    df_tournaments = pd.read_csv('tournamentandteamswithid.csv', dtype={'tournament_id': str})
except FileNotFoundError:
    st.error("The necessary CSV files do not exist in the specified directory.")
    st.stop()

import pandas as pd
from glicko2 import Player
import random

# Step 1: Load the games data
data_path = "games.csv"
games_df = pd.read_csv(data_path)

# Step 2: Filter out irrelevant regions
valid_regions = ['LPL', 'LEC', 'LCK', 'LCS', 'PCS', 'VCS', 'CBLOL', 'LJL', 'LLA']
filtered_games_df = games_df[games_df['teamA_region'].isin(valid_regions) & games_df['teamB_region'].isin(valid_regions)]

# Step 3: Sort the data by event time
filtered_games_df.loc[:, 'eventtime'] = pd.to_datetime(filtered_games_df['eventtime'], format='mixed', utc=True)

sorted_games_df = filtered_games_df.sort_values(by='eventtime')

# Step 4: Initialize parameters and players
cross_regional_weight = 1.1
initial_rd = 350
rd_change_factor = 1.0
rating_reduction_on_roster_change = 150
teams = sorted_games_df['teamA_slug'].unique().tolist()
team_players = {team: Player(rd=initial_rd) for team in teams}

# Step 5: Compute ratings and store in datewise_rating_history dictionary
datewise_rating_history = {team: {} for team in teams}

for index, row in sorted_games_df.iterrows():
    teamA, teamB = row['teamA_slug'], row['teamB_slug']
    scoreA, scoreB = row['teamA_score'], row['teamB_score']
    
    if row['teamA_region'] != row['teamB_region']:
        scoreA *= cross_regional_weight
        scoreB *= cross_regional_weight

    if 'teamA_roster_change' in row and row['teamA_roster_change']:
        team_players[teamA].setRating(team_players[teamA].getRating() - rating_reduction_on_roster_change)
        team_players[teamA].setRd(team_players[teamA].getRd() * rd_change_factor)
    
    if 'teamB_roster_change' in row and row['teamB_roster_change']:
        team_players[teamB].setRating(team_players[teamB].getRating() - rating_reduction_on_roster_change)
        team_players[teamB].setRd(team_players[teamB].getRd() * rd_change_factor)

    team_players[teamA].update_player([team_players[teamB].getRating()], [team_players[teamB].getRd()], [scoreA])
    team_players[teamB].update_player([team_players[teamA].getRating()], [team_players[teamA].getRd()], [scoreB])

    datewise_rating_history[teamA][row['eventtime']] = team_players[teamA].getRating()
    datewise_rating_history[teamB][row['eventtime']] = team_players[teamB].getRating()

# Step 6: Store the latest ratings in a separate dictionary
final_ratings = {team: player.getRating() for team, player in team_players.items()}

# Step 7: Define the get_team_rankings function to fetch the required rankings
def get_team_rankings(teams, use_date_range=False):
    if use_date_range:
        return {team: datewise_rating_history.get(team, {}) for team in teams}
    else:
        return {team: final_ratings.get(team, random.randint(1, 100)) for team in teams}


tabs = ["Team Rankings", "Tournament Rankings", "Global Rankings", "Methodology"]
tab = st.radio("Choose a Tab:", tabs)

if tab == 'Team Rankings':
    # Multiselect dropdown for region selection
    regions = df_teams['region'].unique().tolist() 
    selected_regions = st.multiselect('Select regions:', regions)
    
    # Dropdown for team selection based on selected regions
    if selected_regions:
        teams = df_teams[df_teams['region'].isin(selected_regions)]['slug'].tolist()
    else:
        teams = df_teams['slug'].tolist()

    selected_teams = st.multiselect('Select teams:', teams)
    
    # Checkbox to decide if a date range should be used
    use_date_range = st.checkbox('Use date range')
    
    # Conditionally display date input based on checkbox
    if use_date_range:
        min_date = pd.to_datetime('2020-01-01')  # Replace 'your_min_date_here' with your actual date
        max_date = pd.to_datetime('today')
        
        # Set default start_date and end_date as min_date and max_date
        default_start_date, default_end_date = min_date, max_date 
        
        start_date, end_date = st.date_input('Select a date range:',
                                             format='MM/DD/YYYY', 
                                             value=(default_start_date, default_end_date), 
                                             min_value=min_date, 
                                             max_value=max_date)
        
        # Calculate the number of days between start_date and end_date
        days = (end_date - start_date).days
        
        # Use the calculated days for the DateOffset
        new_date = max_date - pd.DateOffset(days=days)
    else:
        selected_date = st.date_input('Select a date:', format='MM/DD/YYYY')
    
    # Display the rankings if teams are selected
    if st.button('Get Rankings'):
        if not selected_teams and selected_regions:  # If no team is selected but regions are, select all teams from the chosen regions
            selected_teams = teams

        rankings = get_team_rankings(selected_teams, use_date_range)

        if use_date_range:
            # Using Plotly for visualization
            fig = go.Figure()

            # Collect all dates from all teams to ensure they are sorted correctly
            all_dates = set()
            for ranks in rankings.values():
                all_dates.update(ranks.keys())
            sorted_dates = sorted(all_dates)

            for team, ranks in rankings.items():
                scores = [ranks.get(date, None) for date in sorted_dates]  # Use 'get' to ensure missing dates get a 'None' value
                fig.add_trace(go.Scatter(x=sorted_dates, y=scores, mode='lines+markers', name=team, connectgaps=True))  # connectgaps will connect missing data points

            fig.update_layout(title='Team Rankings Over Time',
                              xaxis_title='Date',
                              yaxis_title='ELO Score')

            st.plotly_chart(fig)

        else:
            st.subheader(f"Team Rankings for {selected_date.strftime('%m/%d/%Y')}")
            for team, rank in rankings.items():
                st.write(f"{team}: {rank}")

elif tab == 'Tournament Rankings':
    input_tournament_id = st.text_input('Enter Tournament ID (Optional):')
    
    if input_tournament_id:
        tournament_filtered = df_tournaments[df_tournaments['tournament_id'] == input_tournament_id]
        if not tournament_filtered.empty:
            tournament_teams = tournament_filtered['name'].tolist()
            tournament_rankings = get_team_rankings(tournament_teams)
            displayed_rankings = sorted(tournament_rankings.items(), key=lambda x: x[1], reverse=True)
            
            for i, (tournament, rank) in enumerate(displayed_rankings, 1):
                st.write(f"{i}. {tournament}: {rank}")
                
        else:
            st.write("Invalid Tournament ID or no data available for the entered ID.")
    
    else:
        # Dropdown for year selection
        years = ['All'] + list(range(2020, 2024))
        selected_year = st.selectbox('Select year:', years)

        # If a specific year is selected, filter the tournaments based on that year
        if selected_year != 'All':
            year_filtered_tournaments = df_tournaments[df_tournaments['tournament_slug'].str.contains(str(selected_year))]
        else:
            year_filtered_tournaments = df_tournaments

        # Checkbox for international tournaments
        international = st.checkbox('International tournaments only')

        if international:
            # Filter the dataframe for tournaments that contain 'msi' or 'worlds' in their slug
            international_filtered_tournaments = year_filtered_tournaments[year_filtered_tournaments['tournament_slug'].str.contains('msi|worlds', case=False)]
        else:
            international_filtered_tournaments = year_filtered_tournaments

        # Dropdown for region selection
        regions = ['All'] + international_filtered_tournaments['region'].unique().tolist()
        selected_region = st.selectbox('Select region:', regions)

        if selected_region != 'All':
            # Further filter tournaments based on the selected region
            region_filtered_tournaments = international_filtered_tournaments[international_filtered_tournaments['region'] == selected_region]
        else:
            region_filtered_tournaments = international_filtered_tournaments

        # Allow the user to select a tournament from the filtered list
        selected_tournament_slug = st.selectbox('Select a tournament:', region_filtered_tournaments['tournament_slug'].unique().tolist())

        if selected_tournament_slug:
            # Fetching the teams (from the 'name' column) for the selected tournament
            tournament_teams = region_filtered_tournaments[region_filtered_tournaments['tournament_slug'] == selected_tournament_slug]['name'].tolist()
            
            tournament_rankings = get_team_rankings(tournament_teams)
            displayed_rankings = sorted(tournament_rankings.items(), key=lambda x: x[1], reverse=True)
            
            for i, (tournament, rank) in enumerate(displayed_rankings, 1):
                st.write(f"{i}. {tournament}: {rank}")

if tab == 'Global Rankings':
    total_teams = len(df_teams)

    # Radio button to choose between 'Current' and 'Past' rankings
    ranking_type = st.radio("Choose ranking type:", ["Current", "Past"])
    
    # Checkbox to decide if a date range should be used
    use_date_range = st.checkbox('Use date range') if ranking_type == 'Past' else False

    number_to_display = st.number_input('Number of teams to display:', min_value=1, max_value=total_teams, value=10)

    if ranking_type == 'Current':
        global_rankings = get_team_rankings(df_teams['slug'].tolist())
        displayed_rankings = sorted(global_rankings.items(), key=lambda x: x[1], reverse=True)[:number_to_display]

        for i, (team, rank) in enumerate(displayed_rankings, 1):
            st.write(f"{i}. {team}: {rank}")

    else:  # If 'Past' is selected

        global_rankings = get_team_rankings(df_teams['slug'].tolist(), use_date_range=True)

        if use_date_range:
            min_date = pd.to_datetime('2020-01-01')
            max_date = pd.to_datetime('today')
            default_start_date, default_end_date = min_date, max_date

            start_date, end_date = st.date_input('Select a date range:',
                                                 format='MM/DD/YYYY',
                                                 value=(default_start_date, default_end_date),
                                                 min_value=min_date,
                                                 max_value=max_date)

            # Fetch only the top teams based on the most recent date in their rankings
            most_recent_date = max(max(ranks.keys()) for ranks in global_rankings.values())
            top_teams = sorted(global_rankings.items(), key=lambda x: x[1].get(most_recent_date, 0), reverse=True)[:number_to_display]

            all_data = []
            for team, ranks in top_teams:
                # Convert ranks to a DataFrame and filter by date range
                df = pd.DataFrame(list(ranks.items()), columns=['Date', 'Score'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
                print(df['Date'])
                # Adding a 'Team' column to distinguish data when merged
                df['Team'] = team
                all_data.append(df)

            # Merging all team data into a single DataFrame
            all_teams_df = pd.concat(all_data)

            fig = go.Figure()
            
            fig.update_layout(
                title='Global Rankings Over Time',
                xaxis_title='Date',
                yaxis_title='ELO Score',
                xaxis_type='date'  # This ensures x-axis is treated as continuous time data
            )
            for team in all_teams_df['Team'].unique():
                team_data = all_teams_df[all_teams_df['Team'] == team]

                # Sort the dataframe by date
                team_data = team_data.sort_values(by='Date')

                fig.add_trace(go.Scatter(x=team_data['Date'], y=team_data['Score'], mode='lines+markers', name=team))


            st.plotly_chart(fig)
        else:
            selected_date = st.date_input('Select a date:', format='MM/DD/YYYY')
            global_rankings = {team: ranks[str(selected_date)] for team, ranks in global_rankings.items() if str(selected_date) in ranks}

            displayed_rankings = sorted(global_rankings.items(), key=lambda x: x[1], reverse=True)[:number_to_display]
            for i, (team, rank) in enumerate(displayed_rankings, 1):
                st.write(f"{i}. {team}: {rank}")
    
elif tab == "Methodology":
    st.title("Methodology")
    st.write("This section will explain our methodology in detail. Coming soon...")
