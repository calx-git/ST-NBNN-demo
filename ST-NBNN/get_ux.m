function xs = get_ux(X_list, u, ub)

    sample_num = length(X_list);
    xs = [];
    for i = 1:sample_num,
        X = X_list{i};
        xi = u'*X + ub;

        xs = [xs;xi];
    end