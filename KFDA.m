classdef KFDA
    
    properties
        trainData
        trainClass
        order
        classes
        nClasses
        nObservations
        nFeatures
        nObsPerClas
        priors
        likelihoodFuns
        V
        means
        covariances
        covInv
        covDet
    end
    
    methods
        function obj = KFDA(X, Y, order, regParam, priors)
            %% Kernel Fisher Discriminant Analysis
            % (with polynomial kernel)
            % Arguments
            % - X: the predictors data, a matrix where rows are
            % observations and columns are features
            % - Y: the class of each observation, a cell of strings whose
            % length being equal to the number of rows in X
            % - order: the order of the polynomial expansion, integer
            % - regParam: optional, the regularization parameter. More
            % specifically, the regularization term is equal to the mean
            % value of the kernel matrix times regParam. Default value is
            % 0.25
            % - priors: optional, a structure, each field corresponding to
            % a class and each value coresponding to the prior probability
            % associated with this class. Default value is the empirical
            % probability: number of obs in a class / total number of obs
            
            assert(size(Y, 2) == 1)
            assert(size(Y, 1) == size(X, 1))
            
            obj.trainData = X;
            obj.trainClass = Y;
            obj.order = order;
            plotFigures = false;
            verbose = false;
            obj.classes = sort(unique(obj.trainClass));
            obj.nClasses = length(obj.classes);

            if nargin < 4
                regParam = 0.25;
            elseif nargin == 5
                assert(isa(priors, 'struct') && length(fields(priors)) == obj.nClasses)
            end

            % Assessing the regularity of the number of features
            [obj.nObservations, obj.nFeatures] = size(obj.trainData);

            % Concatenate training activity and generate class vectors (vectors where
            % an entry is one if the pattern belong to this class)
            classVecsTrain = nan(obj.nObservations, obj.nClasses);
            obj.nObsPerClas = nan(1, obj.nClasses);
            for classId = 1:obj.nClasses
                clas = obj.classes{classId};
                classVecsTrain(:, classId) = ismember(obj.trainClass, clas);
                obj.nObsPerClas(classId) = sum(classVecsTrain(:, classId));
            end
            clear classData

            %% II.1) Compute 'gram' (kernel, K) Matrix i.e. dot product of any two high-dim vectors.

            if verbose, fprintf('Compute K...'), end
            K = multinomial_kernel(obj.trainData, obj.trainData, obj.order);
            K = K ./ obj.nObservations;
            if verbose, fprintf(' done!\n'), end

            %% II.2) Compute the 'dual' to the means difference between classes ('M' matrix) and
            %% of pooled covariance ('N' matrix) on a high-dimensional space.

            %First of all we need a 'dual' version of the averages per class.
            %For that means, an auxiliary column of ones and zeros has to be
            %firstly constructed per class.
            MuQ = cell(1, obj.nClasses); % The high dimensional mean of each class
            Mu = zeros(obj.nObservations, 1); % The mean of the means
            for classId = 1 : obj.nClasses
                %See e.g. Scholkopf & Smola 02 ch 14 for justification of the next step.
                MuQ{classId} = K * classVecsTrain(:, classId) ./ obj.nObsPerClas(classId);
                Mu = Mu + MuQ{classId};
            end
            Mu = Mu ./ obj.nClasses;%Just the mean of the class-means (or centroids)

            % Again, see e.g. Scholkopf & Smola 02 ch 14 for justification of the next
            % step. M will represent the "dual" mean-differences (numerator in the FD ration)
            % and N the "dual" pooled covariance.

            M = zeros(obj.nObservations, obj.nObservations);
            N = K * K';

            for classId = 1 : obj.nClasses
                M = M + (MuQ{classId} - Mu) * (MuQ{classId} - Mu)';
                N = N - (obj.nObsPerClas(classId) * (MuQ{classId} * MuQ{classId}'));
            end
            M = M .* (obj.nClasses - 1); % across-class unbiased covariance

            % Regularizing
            mK = abs(mean(K(:)));
            if verbose, disp(['Mean K is ',num2str(mK)]), end

            C = regParam * mK;% The value of 'C' is such that the maximum eigenvector
            %         is proportional to alpha. This is taken as a stable solution.
            %         This value seems ok. C cannot be much smaller than
            %         0.01*mean(mean(K))in MSUA-like data. The bigger, the worse
            %         in-sample classification.

            N = N + C * K; % Emulates SVM-like penalization (i.e. complexity).

            % Extracting eigenvalues and eigenvectors
            [Vtmp, lambda] = eig(M, N);
            lambda = real(diag(lambda));

            % Warning: eigenvalues have to be ordered.
            [~, index] = sort(abs(lambda), 'descend');
            obj.V = Vtmp(:, index);

            projData = K * obj.V;

            if plotFigures
                figure
                hold on
                for classId = 1 : obj.nClasses
                    projDataC = projData(logical(classVecsTrain(:, classId)),1:3);
                    plot3(projDataC(:,1), projDataC(:,2), projDataC(:,3), 'x')
                end
                xlabel('DA 1')
                ylabel('DA 2')
                zlabel('DA 3')
                title('Projected points from the training data')
                hold off
            end
            
            % Reduced projected data to a dimensionality of obj.nClasses - 1,
            % sufficient for separation of the points in obj.nClasses
            redProjData = struct();
            obj.means = struct();
            obj.covariances = struct();
            for classId = 1 : obj.nClasses
                clas = obj.classes{classId};
                redProjData.(clas) = projData(logical(classVecsTrain(:, classId)), 1 : obj.nClasses - 1);
                obj.means.(clas) = mean(redProjData.(clas));
                obj.covariances.(clas) = cov(redProjData.(clas));
            end

            %% Generate priors and likelihood functions

            % Priors
            if nargin < 5
                obj.priors = zeros(1, obj.nClasses);
                for classId = 1 : obj.nClasses
                    obj.priors(classId) = obj.nObsPerClas(classId) / obj.nObservations;
                end
            else
                priorsTmp = priors;
                obj.priors = zeros(1, obj.nClasses);
                for classId = 1 : obj.nClasses
                    clas = obj.classes{classId};
                    obj.priors(classId) = priorsTmp.(clas);
                end
            end


            % Likelihood
            obj.likelihoodFuns = struct();
            warning('error', 'MATLAB:nearlySingularMatrix')
            for classId = 1 : obj.nClasses
                clas = obj.classes{classId};
                try
                    obj.covInv.(clas) = inv(obj.covariances.(clas));
                    obj.covDet.(clas) = det(obj.covariances.(clas));
                catch
                    pvaCov = vpa(obj.covariances.(clas));
                    obj.covInv.(clas) = double(inv(pvaCov));
                    obj.covDet.(clas) = double(det(pvaCov));
                end
                factor = (2 * pi) ^ (-(obj.nClasses - 1) / 2) * (obj.covDet.(clas) ^ -0.5);
                obj.likelihoodFuns.(clas) = @(x) factor * exp(-0.5 * (x - obj.means.(clas)) * obj.covInv.(clas) * (x - obj.means.(clas))');
            end
            
        end
        
        function [results, resPerClass] = predict(obj, X, Y)
            %% Predict the class of new data
            % Arguments are the same as the X and Y in the constructor
            % function
            
            % Checking arguments
            
            assert(size(X, 1) == size(Y, 1))
            [nObsTest, nFeaturesTest] = size(X);
            if obj.nFeatures ~= nFeaturesTest
                error('The number of features must be constant across classes and train/test data')
            end
            
            classVecsTest = nan(nObsTest, obj.nClasses);
            Yid = nan(nObsTest, 1);
            for classId = 1:obj.nClasses
                clas = obj.classes{classId};
                classVecsTest(:, classId) = ismember(Y, clas);
                Yid(logical(classVecsTest(:, classId))) = classId;
            end
            clear classData
            
            %% Computing the test kernel matrix
            K2 = multinomial_kernel(obj.trainData, X, obj.order);
            K2 = K2 ./ obj.nObservations;
            
            %% Projecting data on the discriminant axes
            rpdTest = K2 * obj.V(:, 1 : obj.nClasses - 1); % Reduced projected data test
            
            %% Retrieving the likelihood of each test point
            likelihoods = zeros(nObsTest, obj.nClasses);
            posteriors = zeros(nObsTest, obj.nClasses);

            for patternId = 1 : nObsTest
                for classId = 1 : obj.nClasses
                    clas = obj.classes{classId};
                    likelihoods(patternId, classId) = obj.likelihoodFuns.(clas)(rpdTest(patternId, :));
                end
                posteriors(patternId, :) = likelihoods(patternId,:) .* obj.priors ./ sum(likelihoods(patternId,:) .* obj.priors);
            end
            
            %% Predicting the class of each data point
            [~, predictedClassId] = max(posteriors,[],2);
            
            predictedClass = {nObsTest, 1};
            for classId = 1 : obj.nClasses
                clas = obj.classes{classId};
                [predictedClass{predictedClassId == classId}] = deal(clas);
            end
            predictedClass = predictedClass';
            
            %% Creating the output variables
            results = struct();
            results.likelihood = likelihoods;
            results.posteriors = posteriors;
            results.predictedClassId = predictedClassId;
            results.predictedClass = predictedClass;
            results.errors = results.predictedClassId ~= Yid;
            
            if nargout > 1
                resPerClass = struct();
                for classId = 1 : obj.nClasses
                    clas = obj.classes{classId};
                    classVec = logical(classVecsTest(:, classId));
                    resPerClass.likelihood.(clas) = likelihoods(classVec, :);
                    resPerClass.posteriors.(clas) = posteriors(classVec, :);
                    resPerClass.predictedClassId.(clas) = predictedClassId(classVec, :);
                    resPerClass.predictedClass.(clas) = predictedClass(classVec, :);
                    resPerClass.errors.(clas) = results.errors(classVec, :);
                end
            end
            
        end
        
%         function formatedRes = format_results(results)
%             nObsTest = length(results.errors.(obj.classes{1}));
%             for 
%         end
        
        function projData = project_data(obj, X, nDims)
            %% Project data on the discriminant axes
            % Arguments:
            % - X: same as in the constructor function
            % - nDims: integre, the number of dimensions on which the data
            % is going to be projected
            
            K = multinomial_kernel(obj.trainData, X, obj.order);
            K = K ./ obj.nObservations;
            projData = K * obj.V(:, 1 : nDims);
        end
        
        function KLdiv = KL_divergence(obj)
            %% Symmetrised Kullback - Leibler divergence
            % Returns a symmetrised version of the KL divergence between
            % each pair of class
            
            for classId = 1 : obj.nClasses - 1
                class0 = obj.classes{classId};
                for otherClassId = classId + 1 : obj.nClasses
                    class1 = obj.classes{otherClassId};
                    mu0 = obj.means.(class0)';
                    mu1 = obj.means.(class1)';
                    cov0 = obj.covariances.(class0);
                    cov1 = obj.covariances.(class1);
                    covInv0 = obj.covInv.(class0);
                    covInv1 = obj.covInv.(class1);
                    covDet0 = obj.covDet.(class0);
                    covDet1 = obj.covDet.(class1);

                    KLnew = symmetrised_KL_multiNormal(mu0, cov0, covInv0, covDet0, mu1, cov1, covInv1, covDet1);

                    KLdiv.([class0 '_' class1]) = KLnew;
                end
            end
        end
        
    end
    
end


