import requests
import pandas as pd
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go
from joblib import load
import shap
import os

dataframe = pd.read_csv('data/df.csv')
# Chargement du modèle :
model = load("model.joblib")
print(type(model))

app = Dash(__name__)

# Mise en page du Dashboard
app.layout = html.Div([
    dcc.Input(id='client-id-input', type='number', placeholder='Entrez le Client ID'),
    dcc.Dropdown(id='feature-selector',
                 options=[{'label': col, 'value': col} for col in dataframe.columns if col not in ['TARGET', 'SK_ID_CURR']],
                 placeholder='Sélectionnez une feature à comparer'),
    html.Button('Prédire', id='submit-button', n_clicks=0),
    dcc.Graph(id='probability-scale'),
    html.Div(id='credit-decision'),
    dcc.Graph(id='comparison-graph'),
    dcc.Graph(id='shap-graph')
])

@app.callback(
    [Output('probability-scale', 'figure'),
     Output('credit-decision', 'children'),
     Output('comparison-graph', 'figure'),
     Output('shap-graph', 'figure')],
    [Input('submit-button', 'n_clicks'),
     Input('client-id-input', 'value'),
     Input('feature-selector', 'value')]
)

def update_graph(n_clicks, client_id, selected_feature):
    if n_clicks > 0 and client_id is not None and selected_feature is not None:
        API_URL = "https://credit-api-fwhmc3cwgyg8a6hw.francecentral-01.azurewebsites.net"
        response = requests.get(f"{API_URL}/predict/{client_id}")
        data = response.json()

        # Si une erreur est renvoyée, pas de graphique en cas d'erreur
        if 'error' in data:
            return go.Figure(), data['error'], go.Figure(), go.Figure()

        probabilite = data['probability']
        decision = data['decision']
        seuil = 0.6

        fig = go.Figure()
        # Zone colorée pour les probabilités inférieures au seuil
        fig.add_trace(go.Scatter(
            x=[0, seuil, seuil, 0],
            y=[1, 1, 0, 0],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',  
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Zone : Probabilités inférieures au seuil',
            showlegend=False
        ))

        # Zone colorée pour les probabilités supérieures ou égales au seuil
        fig.add_trace(go.Scatter(
            x=[seuil, 1, 1, seuil],
            y=[1, 1, 0, 0],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',  
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Zone : Probabilités supérieures ou égales au seuil',
            showlegend=False
        ))

        # Créer une ligne pour la colorbar
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0.5, 0.5],  
            mode='lines',
            line=dict(color='black', width=4),  
            name='Colorbar',
            showlegend=False
        ))

        # Seuil sur la colorbar
        fig.add_trace(go.Scatter(
            x=[seuil],
            y=[0.5],  
            mode='markers+text',
            name='Seuil de 0.6',
            marker=dict(color='red', size=10, symbol='circle'),  
            text=["Seuil de 0.6"],
            textposition="bottom center"
        ))

        # Probabilité sur la colorbar
        fig.add_trace(go.Scatter(
            x=[probabilite],
            y=[0.5],  
            mode='markers+text',
            name='Probabilité d\'accord',
            marker=dict(color='lightskyblue', size=10, symbol='circle'),  
            text=[f"Probabilité : {probabilite:.2f}"],
            textposition="bottom center"
        ))

        fig.update_layout(
            title='Échelle de Probabilité d\'Obtention de Crédit',
            xaxis=dict(title='Probabilité', range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1']),
            yaxis=dict(title='', range=[-0.1, 1.1], showticklabels=False),
            height=300,
            showlegend=True,
        )

        decision_text = f"Décision : {decision} (Probabilité : {probabilite:.2f})"

        # Récupération des données du client
        client_data = dataframe[dataframe['SK_ID_CURR'] == client_id]
        if client_data.empty:
             return go.Figure(), "Client ID non trouvé.", go.Figure()

        # Récupération la valeur de la feature locale pour le client
        client_feature_value = client_data[selected_feature].values[0]

        # Récupérer les données de la feature sélectionnée pour l'histogramme
        feature_values = dataframe[selected_feature]

        # Créer l'histogramme de la feature
        comparison_fig = go.Figure()
        comparison_fig.add_trace(go.Histogram(x=feature_values, 
                                              name='Distribution de la feature', 
                                              opacity=0.75, 
                                              marker=dict(color='blue')))

        # Ajouter une ligne verticale pour le client
        comparison_fig.add_trace(go.Scatter(
            x=[client_feature_value, client_feature_value],
            y=[0, max(feature_values.value_counts()) + 1],  # Ajuster en fonction de la distribution
            mode='lines',
            name='Valeur du Client',
            line=dict(color='red', width=4)
        ))

        # Mettre à jour la mise en page du graphique
        comparison_fig.update_layout(
            title=f'Distribution de {selected_feature} et Valeur du Client',
            xaxis_title=selected_feature,
            yaxis_title='Fréquence'
        )

        # Récupération des données du client
        client_data = dataframe[dataframe['SK_ID_CURR'] == client_id]
        if client_data.empty:
             return go.Figure(), "Client ID non trouvé.", go.Figure()
        
        # Récupérer les valeurs SHAP pour le client
        client_features = client_data.drop(['TARGET', 'SK_ID_CURR', 'index'], axis=1)
        assert client_features.shape[1] == 795, "Le nombre de features doit être de 795."

        explainer = shap.Explainer(model)
        shap_values = explainer(client_features.values)

        # Sélectionner les 5 features avec la plus grande valeur absolue de SHAP
        shap_df = pd.DataFrame(shap_values.values, columns=client_features.columns)  # En supposant que 'TARGET' est la dernière colonne
        top_features = shap_df.abs().mean().nlargest(5).index

        # Créer un graphique waterfall
        waterfall_fig = go.Figure(go.Waterfall(
            x=top_features,
            y=shap_df[top_features].iloc[0].values,
            base=0,
            text=[f'Contribution: {value:.2f}' for value in shap_df[top_features].iloc[0].values],
        ))

        # Mettre à jour la mise en page du graphique Waterfall
        waterfall_fig.update_layout(
            title="Contributions des 5 meilleures features",
            xaxis_title="Features",
            yaxis_title="Impact sur la décision",
            showlegend=False
        )

        return fig, decision_text, comparison_fig, waterfall_fig 


    return go.Figure(), "", go.Figure(), go.Figure()


if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
