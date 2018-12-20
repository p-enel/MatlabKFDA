%% Example of how to use the KFDA class
% The data generated here doesn't have the best shape to demonstrate the
% value of KFDA, but at least you have an example of how to use the class

% First execute the first part of the code to train and test the model,
% then execute the second part to visualize the decision boundary

% Generating fake data as two 2nd order polynomials that cannot be
% separated linearly
x1 = linspace(1, 5);
y1 = (x1-3).^2 + rand(1, 100)*2;
x2 = linspace(3, 7);
y2 = -1.5*(x2-5).^2 + rand(1, 100)*2 + 8;

% Concatenate your data into a single matrix (row are observations, columns
% are features or variables)
X = [[x1', y1']; [x2', y2']];
Y = cell(200, 1);
% The class labels must be in a cell
Y(1:100) = {'class1'};
Y(101:200) = {'class2'};

% Creating and training the KFDA object
kfda = KFDA(X, Y, 3);

% Testing the model by generating new data points with the same two
% polynomials
x1test = linspace(1, 5);
y1test = (x1test-3).^2 + rand(1, 100)*2;
x2test = linspace(3, 7);
y2test = -1.5*(x2test-5).^2 + rand(1, 100)*2 + 8;

Xtest = [[x1test', y1test']; [x2test', y2test']];
Ytest = cell(200, 1);
Ytest(1:100) = {'class1'};
Ytest(101:200) = {'class2'};

% Predicting the label of these new points
results = kfda.predict(Xtest, Ytest);

% The following figure shows the new data points are circles in red the
% ones that have been misclassified
plot(x1test, y1test, 'xg');
hold on
plot(x2test, y2test, 'xb');

for iobs = 1:200
    if results.errors(iobs) == 1
        plot(Xtest(iobs, 1), Xtest(iobs, 2), 'ro')
    end
end

%% Here is a way to get an idea of the decision boundary of this model:
xlin = linspace(0, 8);
ylin = linspace(-1, 11);

ys = repmat(ylin, 100, 1);
Xdb = [repmat(xlin', 100, 1) ys(:)];

% Generating fake labels as we are not interested in the performance but
% just the prediction of the model
nprobes = size(Xdb, 1);
Ydb = cell(nprobes, 1);
Ydb(:) = {'class1'};

resdb = kfda.predict(Xdb, Ydb);

figure()
plot(x1, y1, 'xg');
hold on
plot(x2, y2, 'xb');

for iprobe = 1:nprobes
    if strcmp(resdb.predictedClass{iprobe}, 'class1')
        marker = '.g';
    else
        marker = '.b';
    end
    plot(Xdb(iprobe, 1), Xdb(iprobe, 2), marker)
end

% The color of each dot represents the class predicted at that point
% The 'x's correspond to the original data used to train the model')

%% Other methods:

% Project data on the the 'nDims' dimension
projdata = kfda.project_data(Xtest, 1);

% Find the Symmetrised Kullback - Leibler divergence between classes
kldivergence = kfda.KL_divergence();
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
