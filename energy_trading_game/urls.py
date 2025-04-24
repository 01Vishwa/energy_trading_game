from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView, RedirectView
from django.conf import settings
from django.conf.urls.static import static

from households import views as households_views
from trading import views as trading_views
from users import views as user_views  # Your own views and decorator

urlpatterns = [
    # Admin interface
    path("admin/", admin.site.urls),

    # Landing page
    path("", TemplateView.as_view(template_name="index.html"), name="index"),

    # --------------------
    # Authentication paths
    # --------------------
    path("signup/", user_views.signup_view, name="signup"),
    path("login/", user_views.login_view, name="login"),
    path("logout/", user_views.logout_view, name="logout"),

    # -----------------------------------------------------
    # Protected area: only visible if session['user_id'] set
    # -----------------------------------------------------
    path(
        "main/",
        user_views.login_required(households_views.main_view),
        name="main"
    ),

    path(
        "main/report/",
        user_views.login_required(trading_views.energy_report),
        name="energy_report"  # Changed from "report" to "energy_report"
    ),
    path(
        "main/report.html",
        RedirectView.as_view(url="/main/report/", permanent=True)
    ),

    # API & data-viewing endpoints
    path(
        "api/update-trading-data/",
        trading_views.update_energy_trading_collection,
        name="update_trading_data"
    ),
    path(
        "household_data/<str:household_id>/",
        households_views.household_data,
        name="household_data"
    ),
    path("add-household/", households_views.add_household, name="add_household"),
    path("search_households/", households_views.search_households, name="search_households"),
    path(
        "energy_graphs_view/<str:household_id>/",
        households_views.energy_graphs_view,
        name="energy_graphs_view"
    ),
    path("list_households/", households_views.list_households, name="list_households"),
    
    # New URL pattern for the trading game
    path(
        "energy/start-trading-game/",
        trading_views.start_trading_game,
        name="start_trading_game"
    ),
]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)