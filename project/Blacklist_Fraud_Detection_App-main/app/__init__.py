from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
import os
from flask_login import LoginManager


db = SQLAlchemy()
DB_NAME = "database.db"





def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secretkey'
    
    # Ensure instance folder exists
    instance_path = path.join(path.dirname(path.abspath(__file__)), 'instance')
    if not path.exists(instance_path):
        os.makedirs(instance_path)
        
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{path.join(instance_path, DB_NAME)}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    
    # Register CLI commands
    from .commands import create_superuser_command
    
    @app.cli.command('create-superuser')
    def create_superuser():
        """Create a superuser/admin account"""
        create_superuser_command()


    from .views import views
    from .auth import auth


    app.register_blueprint(views, url_prefix = '/')
    app.register_blueprint(auth, url_prefix = '/')


    from .models import User
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))


    return app


def create_database(app):
    if not path.exists('app/instance/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')