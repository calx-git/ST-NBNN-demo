function xs = get_xv(X_list, v, vb)

    sample_num = length(X_list);
    xs = [];
    for i = 1:sample_num,
        X = X_list{i};

        xi = X*v + vb;
        xs = [xs;xi'];
    end