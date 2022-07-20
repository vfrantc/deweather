clear all;
close all;

addpath('libsvm-3.25/matlab/');
addpath('BRISQUE_release');
addpath('niqe_release');
addpath('SSEQ');

out_dir = '../out/';
ds_dir = '../data/';

datasets = {'real-world-images'};
methods  = {'input', 'irr', 'vrg', 'mspfn', 'mprnet', 'hinet', 'qhinet720', 'color_corrected', 'decolor_corrected'};
metrics = {'NIQE', 'BRISQUE', 'SSEQ'};

results = zeros([length(datasets), length(methods), length(metrics)]);
for i = 1:length(datasets)
    disp('----------------------');
    disp(datasets{i});
    disp('----------------------');
    dataset_name = datasets{i};
    
    for j = 1:length(methods)
        disp('-------------------------');
        disp(methods{j});
        disp('-------------------------');
        method_name = methods{j};
        
        images_dir = [out_dir dataset_name '/' method_name];
        
        files = dir(images_dir);
        cnt = 0;
        for k = 1:length(files)
            file = files(k);
            if file.isdir
                continue
            end
            
            filename = [file.folder '/' file.name];
            image = imread(filename);
            disp(size(image));
            image = rgb2gray(image);

            
            %% NIQE
            load niqe_release/modelparameters.mat
            blocksizerow    = 96;
            blocksizecol    = 96;
            blockrowoverlap = 0;
            blockcoloverlap = 0;        
            
            niqe = computequality(image, blocksizerow, blocksizecol, blockrowoverlap, blockcoloverlap, mu_prisparam, cov_prisparam);
            brisque = brisquescore(image);
            sseq = SSEQ(image);
            
            fprintf(1, "%s\t\t niqe: %f brisque: %f sseq: %f", niqe, brisque, sseq);
            results(i, j, 1) = results(i, j, 1) + niqe;
            results(i, j, 2) = results(i, j, 2) + brisque;
            results(i, j, 3) = results(i, j, 3) + sseq;
            cnt = cnt + 1;
        end
        results(i, j, :) = results(i, j, :) / cnt;
    end
end

save results_real