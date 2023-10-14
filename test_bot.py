from .bot import ExampleBot


def test_create():
    """
    Test if the bot can be created
    """
    bot = SneakyBot(id=0, grid_size=(1, 1))
    assert bot is not None
