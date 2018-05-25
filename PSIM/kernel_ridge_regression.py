import numpy as np 
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR,SVC
from sklearn.svm import LinearSVC
from sklearn import linear_model, datasets
from sklearn.decomposition import PCA, RandomizedPCA
from IPython import embed

###################################################################################
#implementation for generating random fourier features and kernel ridge regression. 
######################################################
def median_trick(X, scale):
	#median trick for computing the bandwith for kernel regression.
	N = X.shape[0];
	perm = np.random.choice(N,int(np.ceil(N/2.0)), replace=False);
	dsample = X[perm];
	dmean = np.mean(dsample, axis = 0);
	ddsample = dsample - np.tile(dmean, (dsample.shape[0],1));
	#embed()
	#sigma = np.sqrt(np.median(np.diagonal(np.dot(ddsample, ddsample.T))));
	sigma = np.median(np.linalg.norm(ddsample, axis = 1));

	return scale*sigma; 

def generate_omegas(dim, N):
	#sample N points from normal distribution for random gaussian features. 
	mean = np.zeros(dim);
	cov = np.identity(dim);
	#return the omegas, each row is an i.i.d sample. 
	return np.random.multivariate_normal(mean, cov, N);

def generate_random_features(self, X, bwscale):
	xdim = X.shape[1]; 
	sigma = median_trick(X, bwscale);
	#print sigma
	X_regularized = X / sigma;

	D = int(1.5*xdim); #default: 3 times of the original dim of inputs. 
	Omegas = generate_omegas(xdim, D);

	dp_omega_x = np.dot(X_regularized, Omegas.T);
	sin_omega_x = np.sin(dp_omega_x);
	cos_omega_x = np.cos(dp_omega_x);
	Random_feature = np.zeros([X_regularized.shape[0], 2*D]);
	#combine sin and cos together. 
	Random_feature[:, range(0, 2*D, 2)] = sin_omega_x;
	Random_feature[:, range(1, 2*D, 2)] = cos_omega_x;

	#return [Random_feature, Omegas, sigma]
	tmp_return_feature = np.append(Random_feature, X, axis = 1); 
	return_feature = np.hstack((tmp_return_feature, np.ones((tmp_return_feature.shape[0],1))));
	return [return_feature, Omegas, sigma];


def generate_random_features_with_sigma_omegas(self, X, sigma, Omegas):
	X_regularized = X / sigma;
	dp_omega_x = np.dot(X_regularized, Omegas.T);
	sin_omega_x = np.sin(dp_omega_x);
	cos_omega_x = np.cos(dp_omega_x);
	D = Omegas.shape[0];
	Random_feature = np.zeros([X_regularized.shape[0], 2*D]);
	#combine sin and cos together. 
	Random_feature[:, range(0, 2*D, 2)] = sin_omega_x;
	Random_feature[:, range(1, 2*D, 2)] = cos_omega_x;
	
	#return Random_feature;
	tmp_return_feature = np.append(Random_feature, X, axis = 1);
	return_feature = np.hstack((tmp_return_feature, np.ones((tmp_return_feature.shape[0],1))));
	return return_feature;


def generate_single_random_feature(self, x, sigma, Omegas):
	x_rf = np.zeros(2*Omegas.shape[0]);
	xr = x / sigma;
	pd_x_omega = np.dot(xr, Omegas.T);
	x_rf[range(0,2*Omegas.shape[0], 2)] = np.sin(pd_x_omega);
	x_rf[range(1,2*Omegas.shape[0], 2)] = np.cos(pd_x_omega);
	return x_rf;


class RFF_RidgeRegression(object):

	#initialization
	def __init__(self, Ridge, bwscale = 1.):
		self.ridge = Ridge;
		self.scale = bwscale;
		self.Omegas = np.array([]);
		self.Sigma = 0.0;
		self.params = [self.ridge, self.scale, self.Omegas, self.Sigma];

		self.A = np.array([]); 

	#define functions:
	rff_median_trick = median_trick;
	rff_generate_omegas = generate_omegas;
	rff_generate_random_features = generate_random_features;
	rff_generate_random_features_with_sigma_omegas = generate_random_features_with_sigma_omegas;
	rff_generate_single_random_feature = generate_single_random_feature;

	def initialized(self):
		if self.A.shape[0] == 0:
			return False;
		else:
			return True;

	def initialize(self, x_dim, y_dim):
		tmp_X = np.random.rand(4,x_dim);
		tmp_rfreturn = self.rff_generate_random_features(tmp_X, self.scale);
		rff_x_dim = tmp_rfreturn[0].shape[1];
		self.A = np.zeros((rff_x_dim, y_dim));

	def get_params(self):
		if self.A.shape[0] == 0:
			print "not initialized yet, please call fit function with inputs and outputs first."
			assert False
		return self.A;
	
	def set_params(self, A):
		self.A = A;

	def fit(self, X, Y):
		N = X.shape[0];
		if self.Omegas.shape[0] == 0:
			rfreturn = self.rff_generate_random_features(X, self.scale);
			self.Omegas = rfreturn[1];
			self.Sigma = rfreturn[2];
			Xrf = rfreturn[0];
		else:
			Xrf = self.rff_generate_random_features_with_sigma_omegas(X, self.Sigma, self.Omegas);
			
		#Xrf = rfreturn[0];
		#compute A:
		self.A = np.dot(np.dot(Y.T, Xrf), np.linalg.inv(np.dot(Xrf.T, Xrf) + N * self.ridge*np.identity(Xrf.shape[1]))).T;

	def predict(self, X):
		Xrf = self.rff_generate_random_features_with_sigma_omegas(X, self.Sigma, self.Omegas);
		return np.dot(Xrf, self.A);


class Bagging_RFF_RidgeRegression(object):
	def __init__(self, Ridge, scale = 1., n_estimators = 10, sub_sample = 0.3):
		self.n_estimators = n_estimators;
		self.sub_sample = sub_sample;
		self.ridge = Ridge;
		self.scale = scale;
		self.Omegas = np.array([]);
		self.Sigma = 0.0;
		self.params = [self.ridge, self.scale, self.Omegas, self.Sigma];
		self.A = np.array([]); 

	#define functions:
	rff_median_trick = median_trick;
	rff_generate_omegas = generate_omegas;
	rff_generate_random_features = generate_random_features;
	rff_generate_random_features_with_sigma_omegas = generate_random_features_with_sigma_omegas;
	rff_generate_single_random_feature = generate_single_random_feature;

	def fit(self, X, Y):
		self.A = np.array([]);
		N = X.shape[0];
		rfreturn = self.rff_generate_random_features(X, self.scale);
		Xrf = rfreturn[0];
		self.Omegas = rfreturn[1];
		self.Sigma = rfreturn[2];

		for i in xrange(0, self.n_estimators):
			perm = np.random.choice(Xrf.shape[0], int(self.sub_sample*Xrf.shape[0]), replace = True);
			sub_Xrf = Xrf[perm];
			sub_Y = Y[perm];
			sub_A = np.dot(np.dot(sub_Y.T, sub_Xrf), np.linalg.inv(np.dot(sub_Xrf.T, sub_Xrf) + N * self.ridge*np.identity(sub_Xrf.shape[1]))).T;
			if i == 0:
				self.A = sub_A;
			else:
				self.A = self.A + sub_A;
		self.A = self.A / self.n_estimators;
	
	def predict(self, X):
		Xrf = self.rff_generate_random_features_with_sigma_omegas(X, self.Sigma, self.Omegas);
		return np.dot(Xrf, self.A);





class RidgeRegression(object):

	def __init__(self, Ridge, x_dim = None, y_dim = None):
		self.ridge = Ridge;
		self.A = None;

		if x_dim is not None and y_dim is not None:
			self.initialize(x_dim, y_dim);

	def initialized(self):
		if self.A is None:
			return False;
		else:
			return True;

	def get_params(self):
		return self.A;
	
	def set_params(self,A):
		self.A = A;

	def initialize(self, x_dim, y_dim):
		#self.A = np.random.randn(x_dim, y_dim) * 2.0/np.sqrt(x_dim + y_dim);
		self.A = np.zeros((x_dim+1, y_dim)); #x plus interception. 

	def fit(self, input_X, Y):
		X = np.hstack((input_X, np.ones((input_X.shape[0],1))));
		self.dy = Y.shape[1];
		N = X.shape[0];
		self.A = np.dot(np.dot(Y.T, X), np.linalg.inv(np.dot(X.T,X) + N * self.ridge*np.identity(X.shape[1]))).T;

	def predict(self, input_X):
		X = np.hstack((input_X, np.ones((input_X.shape[0],1))));
		#if self.A.shape[0] == 0:
		#	self.A = np.zeros(X.shape[1], self.dy);
		Yhat = np.dot(X, self.A);
		return Yhat;

class Bagging_RidgeRegression(object):

	def __init__(self, Ridge, n_estimator = 10, sub_sample = 0.3):
		self.ridge = Ridge;
		self.A = np.array([]);
		self.n_estimator = n_estimator;
		self.sub_sample = sub_sample;

	def fit(self, X, Y):
		N = X.shape[0];
		for i in xrange(self.n_estimator):
			perm = np.random.choice(X.shape[0], int(self.sub_sample*X.shape[0]), replace = True);
			sub_X = X[perm];
			sub_Y = Y[perm];
			sub_A = np.dot(np.dot(sub_Y.T, sub_X), np.linalg.inv(np.dot(sub_X.T,sub_X) + N * self.ridge*np.identity(sub_X.shape[1]))).T;
			if i == 0:
				self.A = sub_A;
			else:
				self.A = self.A + sub_A;
		self.A = self.A / self.n_estimator;
	
	def predict(self, X):
		Yhat = np.dot(X, self.A);
		return Yhat;



'''
def Kernel_ridge_regression(X, Y, Ridge, bwscale):
	#input data: X: each row is a sample. 
	#Y: each row is a response of a sample.
	N = X.shape[0]; 
	rfreturn = generate_random_features(X, bwscale);
	Xrf = rfreturn[0]; 
	A = np.dot(np.dot(Y.T, Xrf), np.linalg.inv(np.dot(Xrf.T, Xrf) + N * Ridge*np.identity(Xrf.shape[1])));	
	Model = {'A':A, 'Omegas':rfreturn[1], 'Sigma':rfreturn[2]};
	#compute the training error:
	Y_train = np.dot(Xrf, A.T);
	return [Model, Y_train];

def Kernel_ridge_regression_predict(x, Model):
	xrf = generate_single_random_feature(x, Model['Sigma'], Model['Omegas']);
	return np.dot(xrf, Model['A'].T); 


def Kernel_ridge_regression_batch_predict(X, Model):
	Xrf = generate_random_features_with_sigma_omegas(X, Model['Sigma'], Model['Omegas']);
	return np.dot(Xrf, Model['A'].T);

'''
##############end of implementation of the kernel ridge regression ###################
######################################################################################


#############implementation of kernel SVM/Logistic Regression###################
class RFF_Svm_Logistic(object):

	#initializaiton.
	def __init__(self,  Ridge, bwscale, method_name):
		self.ridge = Ridge;
		self.scale = bwscale;
		self.mathod_name = method_name;
		self.Omegas = np.array([]);
		self.Sigma = 0.0;
		self.params = [self.ridge, self.scale, self.mathod_name, self.Omegas, self.Sigma];

		if method_name == 'SVM':
			self.model = LinearSVC(C = 1./self.ridge, dual = False);
		elif method_name == 'Logistic': 
			self.model = linear_model.LogisticRegression(C=1./self.ridge, dual = False);
		else: 
			print "No such methods avaiable: only SVM and Logistic Regression; now switching to SVM:";
			self.model = LinearSVC(C = 1./self.ridge, dual = False);  #if choose wrong, in default initialize with SVM.


	#define functions
	rff_median_trick = median_trick;
	rff_generate_omegas = generate_omegas;
	rff_generate_random_features = generate_random_features;
	rff_generate_random_features_with_sigma_omegas = generate_random_features_with_sigma_omegas;
	rff_generate_single_random_feature = generate_single_random_feature;

	def fit(self, X, Y):
		rfreturn = self.rff_generate_random_features(X, self.scale);
		Xrf = rfreturn[0];
		self.Omegas = rfreturn[1];
		self.Sigma  = rfreturn[2];
		self.model.fit(Xrf, Y.astype(int));

	def predict(self, X):
		Xrf = self.rff_generate_random_features_with_sigma_omegas(X, self.Sigma, self.Omegas);
		return self.model.predict(Xrf);


class Kernel_SVM(object):
	#this use the exact kernel svm, but use meadian trick to pick the band width. 
	def __init__(self,  Ridge, bwscale):
		self.ridge = Ridge;
		self.scale = bwscale;
		self.Sigma = 0.0;
		self.params = [self.ridge, self.scale, self.Sigma];

		self.model = SVC(kernel = 'rbf');

	def rff_median_trick(self, X, scale):
		#median trick for computing the bandwith for kernel regression.
		N = X.shape[0];
		perm = np.random.choice(N,np.ceil(N/2.0), replace=False);
		dsample = X[perm];
		dmean = np.mean(dsample, axis = 0);
		ddsample = dsample - np.tile(dmean, (dsample.shape[0],1));
		sigma = np.sqrt(np.median(np.diagonal(np.dot(ddsample, ddsample.T))));
		return scale*sigma; 


	def fit(self, X, Y):
		self.Sigma = self.rff_median_trick(X, self.scale); 
		print "band width from meadia trick {}".format(self.Sigma);
		self.model.set_params(C = 1./self.ridge, gamma = 1./self.Sigma);
		self.model.fit(X, Y.astype(int));

	def predict(self, X):
		return self.model.predict(X);



'''
def Kernel_SVM(X, Y, Ridge, bwscale): 
	#input data X (2d matrix): each row is a sample. 
	#input Y (vector): each row is a label. 
	N = X.shape[0]; 
	rfreturn = generate_random_features(X, bwscale);
	Xrf = rfreturn[0]; 

	model = LinearSVC(C = 1.0/Ridge, dual = False);
	model.fit(Xrf, Y.astype(int));

	Model = {'SVM model': model, 'Omegas':rfreturn[1], 'Sigma':rfreturn[2]};
	return Model;

def Kernel_SVM_predict(X, Model):
	Xrf = generate_random_features_with_sigma_omegas(X, Model['Sigma'], Model['Omegas']);
	return Model['SVM model'].predict(Xrf);
'''	

class Dimension_Reduction(object):
	'''
	implementation with PCA or randomized PCA (faster). 
	n_components: number of dimensions that you want to reduce to. 
	random: if true, use randomized PCA.
	'''
	def __init__(self, n_components = 10, random = False): #default: regular PCA.
		self.rand = random;
		if self.rand == False:
			self.pca = PCA(n_components = n_components, whiten = True);
		else:
			self.pca = RandomizedPCA(n_components = n_components, whiten = True);


	def reduce_dimension(self, X):
		'''
		X: is a 2D tensor, Num_of_points * dimnesion.
		'''
		X_hat = self.pca.fit_transform(X);
		return X_hat;

	def restore(self, X_hat):
		X_original = self.pca.inverse_transform(X_hat);
		return X_original;




def extend_context_window(local_features, tw):
	num = local_features.shape[0];
	return_feature = local_features;
	for i in xrange(1, tw+1):
		repfirst = np.tile(local_features[0], (i,1));
		replast = np.tile(local_features[-1], (i,1));
		local_feature_add_first = np.concatenate((repfirst, local_features), axis = 0);
		local_feature_add_last  = np.concatenate((local_features, replast), axis = 0);
		return_feature = np.concatenate((return_feature, local_feature_add_first[0:num], local_feature_add_last[i:]), axis = 1);
	return return_feature;



def context_window(X, tw, y_dim):
	'''
	X is 3D tensor: Num_of_traj * Num_of_step * (x_dim + y_dim), which is exact the same data format for PSIM_smooth. 
	tw: this defines the length of the time window: (2*tw + 1)
	y_dim: the dimension of y (the hints that we will predict).

	Return: new_X is a 3D tensor, the same format as X: Num_of_traj * Num_of_step * ((2kf+1)*x_dim+y_dim)
	Note that we only do time window slice for the observation part, y is unchanged (so, y_dim is unchanged). 

	Call this function right before PSIM_smooth (but should be after the dimension reduction, if one decide to do dim reduction)
	'''
	lists = list(X);
	tw_lists = [];
	for i in xrange(0, len(lists)):
		word = lists[i][:,0:-y_dim]; #do not include the label (y). 
		extended_word = extend_context_window(word, tw);
		tw_lists.append(np.concatenate((extended_word, lists[i][:,-y_dim:]), axis = 1));

	new_X = np.array(tw_lists);
	if new_X.shape[2] != (2*tw+1)*(X.shape[2]-y_dim) + y_dim:
		print "new dimension doesn't match to the expected number. "
		assert False

	return new_X

















