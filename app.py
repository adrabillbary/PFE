import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# Configuration de la page
st.set_page_config(
    page_title="Simulation Lotka-Volterra",
    page_icon="🐺",
    layout="wide"
)

def lotka_volterra_system(t, state, alpha, beta, gamma, delta, K=None):
    """
    Système d'équations différentielles de Lotka-Volterra avec croissance logistique optionnelle
    
    Args:
        t: temps (non utilisé dans ce système autonome)
        state: [x, y] où x = proies, y = prédateurs
        alpha: taux de croissance des proies
        beta: taux de prédation
        gamma: taux de mortalité des prédateurs
        delta: taux de reproduction des prédateurs via consommation
        K: capacité de charge pour les proies (None = croissance exponentielle)
    
    Returns:
        [dx/dt, dy/dt]
    """
    x, y = state
    
    # Croissance des proies avec ou sans capacité de charge
    if K is not None:
        # Modèle logistique : croissance ralentit quand x approche K
        growth_term = alpha * x * (1 - x/K)
    else:
        # Modèle classique : croissance exponentielle
        growth_term = alpha * x
    
    dxdt = growth_term - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])

def euler_method(func, y0, t_span, h, *args):
    """
    Méthode d'Euler explicite pour résoudre un système d'EDO
    
    Args:
        func: fonction définissant le système d'EDO
        y0: conditions initiales
        t_span: [t0, tf] intervalle de temps
        h: pas de temps
        *args: arguments additionnels pour func
    
    Returns:
        t: array des temps
        y: array des solutions
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(1, n):
        # Éviter les valeurs négatives
        y[i-1] = np.maximum(y[i-1], 0.0)
        y[i] = y[i-1] + h * func(t[i-1], y[i-1], *args)
        # Forcer les valeurs positives
        y[i] = np.maximum(y[i], 0.0)
    
    return t, y

def runge_kutta_4(func, y0, t_span, h, *args):
    """
    Méthode de Runge-Kutta d'ordre 4 pour résoudre un système d'EDO
    
    Args:
        func: fonction définissant le système d'EDO
        y0: conditions initiales
        t_span: [t0, tf] intervalle de temps
        h: pas de temps
        *args: arguments additionnels pour func
    
    Returns:
        t: array des temps
        y: array des solutions
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(1, n):
        # Éviter les valeurs négatives
        y[i-1] = np.maximum(y[i-1], 0.0)
        
        k1 = h * func(t[i-1], y[i-1], *args)
        k2 = h * func(t[i-1] + h/2, y[i-1] + k1/2, *args)
        k3 = h * func(t[i-1] + h/2, y[i-1] + k2/2, *args)
        k4 = h * func(t[i-1] + h, y[i-1] + k3, *args)
        
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        # Forcer les valeurs positives
        y[i] = np.maximum(y[i], 0.0)
    
    return t, y

def calculate_oscillation_period(t, x, y):
    """
    Calcule la période moyenne des oscillations en détectant les maxima locaux
    
    Args:
        t: array des temps
        x: populations de proies
        y: populations de prédateurs
    
    Returns:
        period_x: période moyenne pour les proies
        period_y: période moyenne pour les prédateurs
    """
    def find_peaks(signal):
        """Trouve les indices des maxima locaux"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return peaks
    
    peaks_x = find_peaks(x)
    peaks_y = find_peaks(y)
    
    period_x = None
    period_y = None
    
    if len(peaks_x) > 1:
        periods_x = [t[peaks_x[i+1]] - t[peaks_x[i]] for i in range(len(peaks_x)-1)]
        period_x = np.mean(periods_x)
    
    if len(peaks_y) > 1:
        periods_y = [t[peaks_y[i+1]] - t[peaks_y[i]] for i in range(len(peaks_y)-1)]
        period_y = np.mean(periods_y)
    
    return period_x, period_y

def create_plots(t, sol, alpha, beta, gamma, delta, K=None, show_vector_field=False):
    """
    Crée les graphiques d'évolution temporelle et de phase
    
    Args:
        t: array des temps
        sol: solution du système [x, y]
        alpha, beta, gamma, delta: paramètres du modèle
        K: capacité de charge (None si modèle classique)
        show_vector_field: afficher le champ de vecteurs sur le diagramme de phase
    
    Returns:
        fig: figure matplotlib
    """
    x, y = sol[:, 0], sol[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique d'évolution temporelle
    ax1.plot(t, x, 'b-', label='Proies (x)', linewidth=2)
    ax1.plot(t, y, 'r-', label='Prédateurs (y)', linewidth=2)
    
    # Ajouter ligne de capacité de charge si modèle logistique
    if K is not None:
        ax1.axhline(y=K, color='blue', linestyle='--', alpha=0.7, label=f'Capacité de charge (K={K})')
    
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Population')
    title = 'Évolution temporelle des populations'
    if K is not None:
        title += ' (Modèle logistique)'
    else:
        title += ' (Modèle classique)'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Diagramme de phase
    if show_vector_field:
        # Créer une grille pour le champ de vecteurs
        x_max = max(np.max(x) * 1.1, 10)
        y_max = max(np.max(y) * 1.1, 10)
        
        # Grille plus dense pour un meilleur rendu
        x_grid = np.linspace(0.1, x_max, 15)
        y_grid = np.linspace(0.1, y_max, 12)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Calculer les dérivées à chaque point de la grille
        DX = np.zeros_like(X_grid)
        DY = np.zeros_like(Y_grid)
        
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                state = np.array([X_grid[i, j], Y_grid[i, j]])
                derivatives = lotka_volterra_system(0, state, alpha, beta, gamma, delta, K)
                DX[i, j] = derivatives[0]
                DY[i, j] = derivatives[1]
        
        # Normaliser les vecteurs pour une meilleure visualisation
        M = np.sqrt(DX**2 + DY**2)
        M[M == 0] = 1  # Éviter la division par zéro
        DX_norm = DX / M
        DY_norm = DY / M
        
        # Afficher le champ de vecteurs
        ax2.quiver(X_grid, Y_grid, DX_norm, DY_norm, M, 
                  scale=20, alpha=0.6, cmap='viridis', 
                  width=0.003, headwidth=3, headlength=4)
    
    # Trajectoire de phase
    ax2.plot(x, y, 'g-', linewidth=2.5, alpha=0.8, label='Trajectoire')
    ax2.plot(x[0], y[0], 'go', markersize=8, label='Point initial')
    ax2.plot(x[-1], y[-1], 'ro', markersize=8, label='Point final')
    
    # Point d'équilibre théorique (si pas de capacité de charge)
    if K is None:
        x_eq = gamma / delta
        y_eq = alpha / beta
        ax2.plot(x_eq, y_eq, 'k*', markersize=12, label=f'Équilibre ({x_eq:.1f}, {y_eq:.1f})')
    
    ax2.set_xlabel('Proies (x)')
    ax2.set_ylabel('Prédateurs (y)')
    title = 'Diagramme de phase'
    if show_vector_field:
        title += ' avec champ de vecteurs'
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, None)
    
    plt.tight_layout()
    return fig

def main():
    # Titre principal
    st.title("🐺 Simulation Lotka-Volterra: Dynamique Proie-Prédateur")
    
    # Section explicative avec équations
    st.markdown("## 📐 Modèle mathématique")
    st.markdown("""
    Le système de Lotka-Volterra est décrit par les équations différentielles suivantes :
    """)
    
    # Switch pour choisir le modèle
    use_logistic = st.checkbox("🔄 Activer la croissance logistique (capacité de charge)", value=False)
    
    if use_logistic:
        st.latex(r'''
        \begin{cases}
        \frac{dx}{dt} = \alpha x \left(1 - \frac{x}{K}\right) - \beta x y \\
        \frac{dy}{dt} = \delta x y - \gamma y
        \end{cases}
        ''')
        
        st.markdown("""
        **Modèle avec capacité de charge** - La croissance des proies est limitée par la capacité de charge K.
        """)
    else:
        st.latex(r'''
        \begin{cases}
        \frac{dx}{dt} = \alpha x - \beta x y \\
        \frac{dy}{dt} = \delta x y - \gamma y
        \end{cases}
        ''')
        
        st.markdown("""
        **Modèle classique** - Croissance exponentielle illimitée des proies en l'absence de prédateurs.
        """)
    
    st.markdown("""
    Où :
    - **x** : population des proies
    - **y** : population des prédateurs
    - **α** : taux de croissance des proies
    - **β** : taux de prédation
    - **δ** : taux de reproduction des prédateurs via consommation
    - **γ** : taux de mortalité des prédateurs
    """ + ("- **K** : capacité de charge maximale pour les proies" if use_logistic else ""))
    
    # Barre latérale pour les paramètres
    st.sidebar.header("⚙️ Paramètres de simulation")
    
    # Paramètres biologiques
    st.sidebar.subheader("Paramètres biologiques")
    alpha = st.sidebar.slider("α - Taux de croissance des proies", 0.1, 2.0, 0.55, 0.01)
    beta = st.sidebar.slider("β - Taux de prédation", 0.01, 0.1, 0.028, 0.001)
    delta = st.sidebar.slider("δ - Taux de reproduction des prédateurs", 0.01, 0.1, 0.021, 0.001)
    gamma = st.sidebar.slider("γ - Taux de mortalité des prédateurs", 0.1, 2.0, 0.75, 0.01)
    
    # Capacité de charge (seulement si croissance logistique activée)
    K = None
    if use_logistic:
        K = st.sidebar.slider("K - Capacité de charge des proies", 20, 200, 100, 5)
    
    # Conditions initiales
    st.sidebar.subheader("Conditions initiales")
    x0 = st.sidebar.slider("x₀ - Population initiale des proies", 1, 100, 30)
    y0 = st.sidebar.slider("y₀ - Population initiale des prédateurs", 1, 20, 4)
    
    # Paramètres numériques
    st.sidebar.subheader("Paramètres numériques")
    h = st.sidebar.slider("h - Pas de temps", 0.01, 0.5, 0.1, 0.01)
    T = st.sidebar.slider("T - Durée de simulation", 10, 200, 100, 5)
    method = st.sidebar.selectbox("Méthode numérique", ["Runge-Kutta 4", "Euler"])
    
    # Option pour afficher le champ de vecteurs
    show_vectors = st.sidebar.checkbox("🔍 Afficher le champ de vecteurs", value=False, help="Visualise la direction du flux du système dynamique sur le diagramme de phase")
    
    # Simulation
    y0_vec = np.array([x0, y0])
    t_span = [0, T]
    
    if method == "Runge-Kutta 4":
        t, sol = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, h, alpha, beta, gamma, delta, K)
    else:
        t, sol = euler_method(lotka_volterra_system, y0_vec, t_span, h, alpha, beta, gamma, delta, K)
    
    # Affichage des graphiques
    st.markdown("## 📊 Résultats de la simulation")
    fig = create_plots(t, sol, alpha, beta, gamma, delta, K, show_vectors)
    st.pyplot(fig)
    
    # Statistiques
    x_vals, y_vals = sol[:, 0], sol[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Statistiques des proies (x)")
        st.metric("Maximum", f"{np.max(x_vals):.2f}")
        st.metric("Minimum", f"{np.min(x_vals):.2f}")
        st.metric("Moyenne", f"{np.mean(x_vals):.2f}")
    
    with col2:
        st.markdown("### 🦅 Statistiques des prédateurs (y)")
        st.metric("Maximum", f"{np.max(y_vals):.2f}")
        st.metric("Minimum", f"{np.min(y_vals):.2f}")
        st.metric("Moyenne", f"{np.mean(y_vals):.2f}")
    
    # Calcul des périodes
    period_x, period_y = calculate_oscillation_period(t, x_vals, y_vals)
    
    st.markdown("### 🔄 Analyse des oscillations")
    col3, col4 = st.columns(2)
    
    with col3:
        if period_x is not None:
            st.metric("Période moyenne (proies)", f"{period_x:.2f} unités de temps")
        else:
            st.metric("Période moyenne (proies)", "Non détectable")
    
    with col4:
        if period_y is not None:
            st.metric("Période moyenne (prédateurs)", f"{period_y:.2f} unités de temps")
        else:
            st.metric("Période moyenne (prédateurs)", "Non détectable")
    
    # Export CSV
    st.markdown("### 💾 Export des données")
    
    # Création du DataFrame
    df = pd.DataFrame({
        'Temps': t,
        'Proies_x': x_vals,
        'Predateurs_y': y_vals
    })
    
    # Bouton de téléchargement
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    # Nom du fichier avec indication du modèle utilisé
    model_type = "logistique" if K is not None else "classique"
    filename = f"lotka_volterra_{model_type}_alpha{alpha}_beta{beta}_gamma{gamma}_delta{delta}"
    if K is not None:
        filename += f"_K{K}"
    filename += ".csv"
    
    st.download_button(
        label="📥 Télécharger les données (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )
    
    # Aperçu des données
    with st.expander("👀 Aperçu des données"):
        st.dataframe(df.head(10))
    
    # Section "À propos"
    st.markdown("## 📚 À propos du modèle Lotka-Volterra")
    
    with st.expander("ℹ️ Explication détaillée"):
        st.markdown("""
        ### Historique
        Le modèle de Lotka-Volterra, développé indépendamment par Alfred Lotka (1925) et Vito Volterra (1926), 
        est l'un des premiers modèles mathématiques décrivant la dynamique des populations en interaction.
        
        ### Principe
        Ce modèle décrit l'évolution de deux populations :
        - **Les proies** qui croissent exponentiellement en l'absence de prédateurs
        - **Les prédateurs** qui dépendent entièrement des proies pour survivre
        
        ### Équations
        - **dx/dt = αx - βxy** : Les proies croissent à un taux α mais sont consommées proportionnellement au produit des deux populations
        - **dy/dt = δxy - γy** : Les prédateurs se reproduisent proportionnellement à leur consommation de proies mais meurent à un taux γ
        
        ### Propriétés remarquables
        - **Conservation de l'énergie** : Le système possède une intégrale première (fonction conservée)
        - **Oscillations périodiques** : Les populations oscillent de manière périodique
        - **Point d'équilibre** : Le système possède un point d'équilibre stable en (γ/δ, α/β)
        
        ### Applications
        Ce modèle, bien que simplifié, reste fondamental pour comprendre :
        - Les cycles de population dans la nature (lynx-lièvres au Canada)
        - La dynamique des épidémies
        - Les modèles économiques de concurrence
        - L'écologie théorique
        
        ### Limitations du modèle classique
        - Absence de capacité de charge pour les proies
        - Croissance exponentielle irréaliste des proies
        - Pas de saturation dans la prédation
        - Modèle déterministe sans bruit environnemental
        
        ### Améliorations avec la croissance logistique
        Le modèle avec capacité de charge K introduit une limitation réaliste :
        - **Croissance ralentie** : Quand x approche K, la croissance des proies diminue
        - **Stabilisation** : Les oscillations peuvent s'amortir vers un point d'équilibre
        - **Réalisme écologique** : Reproduit la limitation des ressources dans la nature
        - **Dynamiques diverses** : Selon les paramètres, on peut observer différents comportements (oscillations amorties, point fixe stable, etc.)
        
        #### Comparaison des modèles
        - **Modèle classique** : Oscillations perpétuelles d'amplitude constante
        - **Modèle logistique** : Peut produire des oscillations amorties convergeant vers un équilibre stable
        
        ### Champ de vecteurs
        Le champ de vecteurs sur le diagramme de phase montre :
        - **Direction du flux** : Chaque flèche indique la direction d'évolution du système à ce point
        - **Vitesse d'évolution** : La couleur indique l'intensité (vitesse) du changement
        - **Points d'équilibre** : Zones où les vecteurs convergent ou divergent
        - **Basins d'attraction** : Régions qui évoluent vers le même comportement asymptotique
        
        Cette visualisation aide à comprendre pourquoi certaines trajectoires sont stables et d'autres instables.
        """)

if __name__ == "__main__":
    main()
