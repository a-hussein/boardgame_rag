from boardgame_rag.data_gen import _rand_text

print(
    _rand_text(
        name="Catan", 
        mechs=["Deck Building","Area Control","Worker Placement"], 
        cats=["Economic","Card Game","Eurogame"], 
        weight=2.1,
        t=45
    )
)
