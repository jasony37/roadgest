from main import main
import cProfile

cProfile.run('main()', 'main.profile')