import LSTMAttnSequential
from LSTMAttnSequential import space, objective_fc, space_epoch, objective_findEpoch
from LSTMAttnHybrid import space as spaceA
from LSTMAttnHybrid import objective_fcA
from skopt import gp_minimize

choice = 2

if choice == 0:
    res1 = gp_minimize(
        func=objective_fc,
        dimensions=space,
        n_calls=30, 
        random_state=42
    )   

    print("Best score=%.4f" % res1.fun)
    print("Best parameters:")
    print("hidden_dim=%d, num_head=%d, lr=%.6f, epoch_num=%d"
        % (res1.x[0], res1.x[1], res1.x[2], res1.x[3]))
        # Best score=0.0733
        # Best parameters:
        # hidden_dim=132, num_head=2, lr=0.000829, epoch_num=17
    
if choice == 1:
    res2 = gp_minimize(
        func=objective_fcA,
        dimensions=spaceA,
        n_calls=30,
        random_state=42
    )

    print("Best score=%.4f" % res2.fun)
    print("Best parameters:")
    print("hidden_dim=%d, lr=%.6f, epoch_num=%d, tolerance=%.4f"
        % (res2.x[0], res2.x[1], res2.x[2], res2.x[3]))
        # Best score=0.6990
        # Best parameters:
        # hidden_dim=70, lr=0.010000, epoch_num=30, tolerance=9.0455

# As observed, LstmAttnSequential model provides more accuracy than the other model.

# if choice == 2:
#     res1 = gp_minimize(
#         func=objective_findEpoch,
#         dimensions=space_epoch,
#         n_calls=10, 
#         random_state=42
#     )   

#     print("Best score=%.4f" % res1.fun)
#     print("Best parameters:")
#     print("epoch_num=%d"
#         % (res1.x[0]))

# Finding with manually is more time efficient.




