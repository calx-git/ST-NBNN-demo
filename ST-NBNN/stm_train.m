function [accNBNNTrain accNBNNTest accTrains, accTests] = stm_train(X, Y, X_test, Y_test, lambda, penelty,  weight)

    class_num = length(unique(Y));
    T = 1;

    X_list = X{1};
    X_list_test = X_test{1};

    numSampleTrain = length(X_list);
    numSampleTest  = length(X_list_test);

    X0 = X_list{1};
    [row, col] = size(X0);

    yss     = zeros(numSampleTrain * T * 2, class_num);
    yssTest = zeros(numSampleTest  * T * 2, class_num);

    meanTrain = zeros(1,numSampleTrain);
    meanTest  = zeros(1,numSampleTest);

    sumMatTrain = zeros(numSampleTrain,class_num);
    sumMatTest  = zeros(numSampleTest, class_num);

    for samIdx = 1:numSampleTrain

        X_list = [];
        preSum = [];

        for c = 1:class_num
            X_list = [X_list      X{c}{samIdx}(:)];
            sumMatTrain(samIdx,c) = sum(X{c}{samIdx}(:));
        end

        meanTrain(samIdx) = mean(X_list(:));
    end

    for samIdx = 1:numSampleTest

        X_list_test = [];

        for c = 1:class_num
            X_list_test = [X_list_test X_test{c}{samIdx}(:)];
            sumMatTest(samIdx,c) = sum(X_test{c}{samIdx}(:));
        end

        meanTest(samIdx) = mean(X_list_test(:));
    end



    for c = 1:class_num

        X_list = X{c};
        X_list_test = X_test{c};

        % centralization
        for samIdx = 1:numSampleTrain,
            X_list{samIdx} = -X_list{samIdx} + meanTrain(samIdx);
        end

        for samIdx = 1:numSampleTest,
            X_list_test{samIdx} = -X_list_test{samIdx} + meanTest(samIdx);
        end

        v = ones(col, 1);
        vb = 0;

        Yc = -1*ones(size(Y));
        cidx = find(Y(:, 1) == c);
        Yc(cidx, 1) = 1;

        j = 1;


        % ST-NBNN
        for i = 1:T

            % Spatial Weight Learning
            xs = get_xv(X_list, v, vb);
            [u, ub] = learn_weight(xs, Yc, ['-c ' num2str(penelty) ' -s 2 -w1 ' num2str(weight) ' -w-1 1']);
            ys = xs*u+ub; % predict

            yss( ((j-1)*numSampleTrain+1) : (j)*numSampleTrain, c ) = ys;
            xsTest = get_xv(X_list_test, v, vb);
            ysTest = xsTest*u+ub;
            yssTest( ((j-1)*numSampleTest+1) : (j)*numSampleTest, c ) = ysTest;

            j = j + 1;

            
            % Temporal Weight Learning
            xs = get_ux(X_list, u, ub);
            [v, vb] = learn_weightT(xs, Yc, penelty, 1, weight);
            ys = xs*v+vb; % predict

            yss(((j-1)*numSampleTrain + 1) : (j)*numSampleTrain, c ) = ys;
            xsTest = get_ux(X_list_test, u, ub);
            ysTest = xsTest*v+vb;
            yssTest( ((j-1)*numSampleTest + 1) : (j)*numSampleTest, c ) = ysTest;

            j = j + 1;


        end

    end


    % NBNN Accuracy
    [value pred_label] = min(sumMatTrain,[],2);
    correct_idx = find(Y == pred_label);
    accNBNNTrain = length(correct_idx)/length(Y);

    [value pred_label] = min(sumMatTest,[],2);
    correct_idx = find(Y_test == pred_label);
    accNBNNTest = length(correct_idx)/length(Y_test);


    accTrains = [];
    accTests  = [];

    for i = 1 : 2*T

        ysRound = yss((i-1)*numSampleTrain+1 : i*numSampleTrain, :);
        [value pred_label] = max(ysRound,[],2);
        correct_idx = find(Y == pred_label);
        accsTrain = length(correct_idx)/length(Y);
        accTrains = [accTrains; accsTrain];

        ysRound = yssTest((i-1)*numSampleTest+1 : i*numSampleTest, :);
        [value pred_label] = max(ysRound,[],2);
        correct_idx = find(Y_test == pred_label);
        accsTest = length(correct_idx)/length(Y_test);
        accTests = [accTests; accsTest];

    end