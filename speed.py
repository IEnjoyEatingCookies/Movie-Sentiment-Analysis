import d_m as dm

while True:
    inp = input('Word:')
    
    vec = dm.data_to_input(inp, 1)[0][0]
    print(vec)
    print(vec.shape)


    word = dm.vec2word(vec, 2)

    print(word)
