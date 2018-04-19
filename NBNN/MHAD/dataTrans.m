clear all;
close all;
clc;

fidtrain = fopen('train.txt', 'r');
line = 'begin';
clsList = [];
while ischar(line)
    line  = fgets(fidtrain);
    if length(line) > 1
        clsIdx = str2num(line(2:3));
        
        if ~ismember(clsIdx,clsList)
            clsList = [clsList clsIdx];
        end
    end
end
clsList
fclose(fidtrain);


NUM_CLASS = length(clsList);

for treeIdx = 0:(NUM_CLASS-1)
    
    train_X_list = {};
    test_X_list  = {};
    train_Y = [];
    test_Y  = [];

    trainCount = 1;
    testCount  = 1;

    rootDir = 'all';


    fidtrain = fopen('train.txt', 'r');

    line = 'begin';

    while ischar(line)
        line  = fgets(fidtrain);
        if length(line) > 1
            video = [line(1:11) '_' num2str(treeIdx)];

            stm = load([rootDir '/' video]);
    %         stm = rowSelect(stm);

            train_X_list{trainCount,1} = stm;
            clsIdx = str2num(video(2:3));

            train_Y = [train_Y; find(clsIdx == clsList)];
            trainCount = trainCount + 1;
            
            disp([video ' ' num2str(find(clsIdx == clsList))]);
        end
    end

    fclose(fidtrain);



    fidtest  = fopen('test.txt', 'r');
    line = 'begin';

    while ischar(line)
        line  = fgets(fidtest);
        if length(line)>1
            video = [line(1:11) '_' num2str(treeIdx)];

            stm = load([rootDir '/' video]);
    %         stm = rowSelect(stm);

            test_X_list{testCount,1} = stm;
            clsIdx = str2num(video(2:3));

            test_Y = [test_Y; find(clsIdx == clsList)];
            testCount = testCount + 1;
            
            disp([video ' ' num2str(find(clsIdx == clsList))]);
        end
    end

    fclose(fidtest);


    save(['data' '_' num2str(treeIdx) '.mat'], 'train_X_list', 'train_Y', 'test_X_list', 'test_Y');
    
end

