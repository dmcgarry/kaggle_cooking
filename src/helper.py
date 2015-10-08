from sklearn.base import BaseEstimator, TransformerMixin

class ExtractRecipe():
    """ 
    Class that extracts recipe information from JSON.
    """
    def __init__(self,json):
        self.recipe_id = self.set_id(json)
        self.cuisine = self.set_cuisine(json)
        self.ingredients = self.set_ingredients(json)
        self.ingredient_count = len(self.ingredients)
    def __str__(self):
        return "ID: %s\nCuisine: %s\nIngredients: %s\nNumber of Ingredients: %s" % (self.recipe_id, self.cuisine,', '.join(self.ingredients),self.ingredient_count)
    def set_id(self,json):
        """
        Method that sets the recipe id.
        """
        try:
            return json['id']
        except KeyError:
            return '-99'
    def set_cuisine(self,json):
        """
        Method that sets the recipe cuisine.
        """
        try:
            return json['cuisine']    
        except KeyError:
            return ''
    def set_ingredients(self,json):
        """
        Method that sets the recipe ingredients.
        """
        try:
            return json['ingredients']
        except KeyError:
            return []
    def clean_ingredient(self,s):
    	"""
    	Method that returns a cleaned up version of the entered ingredient.
    	"""
    	return ''.join(x for x in s if x.isalnum())
    def get_train(self):
        """
        Method that returns a dictionary of data for the training set.
        """
        return {
            'cuisine':self.cuisine,
            'ingredients':', '.join([self.clean_ingredient(x) for x in self.ingredients]),
            'ingredient_count':self.ingredient_count
        }
    def get_predict(self):
        """
        Method that returns a dictionary of data for predicting recipes.
        """
        return {
            'id':self.recipe_id,
            'ingredients':', '.join([self.clean_ingredient(x) for x in self.ingredients]),
            'ingredient_count':self.ingredient_count
        }   

class IngredientModel():
	"""
	Class that stores an ingredient to cuisine model.
	"""
	def __init__(self,model):
		self.model = model
	def predict(self,X):
		from pandas import Series
		from operator import add
		return X.ingredients.str.split(',? ').apply(lambda recipe: Series(reduce(add,[self.model.predict_proba([x]) for x in recipe])[0]/len(recipe)))

class TextModel():
	"""
	Class that stores and a simple weighted average of two text-based individual cuisine models.
	"""
	def __init__(self,a_model,b_model):
		self.a_model = a_model
		self.b_model = b_model
		self.a_weight = 0.5
		self.b_weight = 0.5
	def set_weights(self,a_weight,b_weight):
		self.a_weight = a_weight
		self.b_weight = b_weight
	def blend(self,a_pred,b_pred):
		return a_pred*self.a_weight + b_pred*self.b_weight
	def predict(self,X):
		a_pred = self.a_model.predict_proba(X)[:,1]
		b_pred = self.b_model.predict_proba(X)[:,1]
		return self.blend(a_pred,b_pred)

class RecipeModel():
	"""
	Class that stores the models needed to predict the type of cuisine based on a list of ingredients.
	"""
	def __init__(self,ingred_model,text_models,recipe_model_a,recipe_model_b,encoder):
		self.ingred_model = ingred_model
		self.text_models = text_models
		self.recipe_model_a = recipe_model_a
		self.recipe_model_b = recipe_model_b
		self.recipe_weight_a = 0.5
		self.recipe_weight_b = 0.5
		self.score = 0.0
		self.encoder = encoder
	def __str__(self):
		return "\nRecipe Model\nBlended Accuracy: %0.5f\nModel A Weight: %0.2f\nModel B Weight: %0.2f" % (self.score, self.recipe_weight_a, self.recipe_weight_b)
	def set_weights(self,pred_a,pred_b,target):
		from sklearn.metrics import accuracy_score
		for w in zip(range(1,100,1),range(99,0,-1)):
				score = accuracy_score(target,(pred_a*w[0]/100.0+pred_b*w[1]/100.0).argmax(1))
				if score > self.score:
					self.recipe_weight_a = w[0]/100.0
					self.recipe_weight_b = w[1]/100.0
					self.score = score
	def predict_kaggle(self,X,prob=False):
		# add average ingredient scores for each cuisine
		X = X.join(self.ingred_model.predict(X))
		# add cuisine based text models
		for v in self.text_models.keys():
			X['pred_text_'+v] = self.text_models[v].predict(X.ingredients)
		# make prediction for recipe model
		pred_a = self.recipe_model_a.predict_proba(X)
		pred_b = self.recipe_model_b.predict_proba(X)
		pred = pred_a*self.recipe_weight_a + pred_b*self.recipe_weight_b
		if prob:
			return pred
		else:
			return self.encoder.inverse_transform(pred.argmax(1))
	def predict(self,json_list,prob=False):
		"""
		Return: The predicted cuisine for the list of recipes. 
		Params:
			* json_list (List of Dicts): The list of JSON recipes seeking cuisine predictions. 
			* prob: (Boolean) If the output should be the predicted probability across all cuisines or the best guess label. Defaults to False.	
		Doctest:
		>>> json_list = [
		...     {
		...             'id':1,
		...             'ingredients': ['pork, black beans, avocado, orange, cumin, salt, cinnamon']
		...     },
		...     {
		...             'id':2,
		...             'ingredients': ['pasta, basil, pine nuts, olive oil, parmesan cheese, garlic']
		...     },
		...     {
		...             'id':3,
		...             'ingredients': ['tumeric, red lentils, naan, garam masala, onions, sweet potatoes']
		...     }
		... ]
		>>> recipe_model.predict(json_list)
		array([u'mexican', u'italian', u'indian'], dtype=object)
		>>> recipe_model.predict(json_list,prob=True)
		array([[  3.30066887e-03,   1.99567227e-06,   1.69680381e-03,
				  1.05174537e-05,   2.03085874e-05,   3.29112324e-03,
				  1.15989352e-06,   4.94386325e-03,   3.40759867e-06,
				  5.00931910e-03,   1.64480210e-03,   1.65024605e-03,
				  7.54782889e-07,   9.33262747e-01,   5.83895764e-07,
				  3.17332176e-06,   2.34691288e-02,   1.01873500e-02,
				  6.56450009e-03,   4.93754585e-03],
			   [  1.48577580e-05,   4.19468051e-06,   1.68586413e-03,
				  1.27146558e-05,   8.20049665e-06,   1.16352625e-02,
				  3.69138042e-02,   3.23459014e-05,   1.23802178e-04,
				  5.19810867e-01,   1.16872025e-05,   1.17136206e-04,
				  6.88425371e-06,   4.24320803e-01,   1.23880210e-05,
				  1.07642463e-05,   3.49156883e-03,   1.75572406e-03,
				  2.47755352e-05,   6.35768902e-06],
			   [  3.28104312e-03,   1.04083038e-05,   1.76803405e-06,
				  1.64068986e-03,   9.84115643e-03,   1.66666236e-03,
				  1.31338270e-02,   6.97641581e-01,   3.29681417e-03,
				  1.31455590e-02,   2.93287824e-06,   7.81646840e-02,
				  3.98645284e-07,   4.43144651e-02,   1.05028641e-01,
				  2.53745611e-06,   2.05663297e-02,   1.67476028e-03,
				  6.58238004e-03,   3.37299002e-06]])
		"""
		from pandas import DataFrame
		# extract features from JSON
		X = DataFrame([ExtractRecipe(x).get_predict() for x in json_list])
		# add average ingredient scores for each cuisine
		X = X.join(self.ingred_model.predict(X))
		# add cuisine based text models
		for v in self.text_models.keys():
			X['pred_text_'+v] = self.text_models[v].predict(X.ingredients)
		# make prediction for recipe model
		pred_a = self.recipe_model_a.predict_proba(X)
		pred_b = self.recipe_model_b.predict_proba(X)
		pred = pred_a*self.recipe_weight_a + pred_b*self.recipe_weight_b
		if prob:
			return pred
		else:
			return self.encoder.inverse_transform(pred.argmax(1))

class ShuffleText(BaseEstimator, TransformerMixin):
    """
    Shuffle the order of ingredients in the recipe.
    """
    def fit(self, x, y=None):
        return self
    def transform(self, series):   	
        return series.str.split(', ?').apply(self.shuffText)        
    def shuffText(self, x):
    	from random import shuffle
    	shuffle(x)
    	return ', '.join(x)
    	
class VarSelect(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        return df[self.keys]

def loadTrainSet(dir='../data/train.json'):
	"""
	Read in JSON to create training set.
	"""
	import json
	from pandas import DataFrame, Series
	from sklearn.preprocessing import LabelEncoder
	X = DataFrame([ExtractRecipe(x).get_train() for x in json.load(open(dir,'rb'))])
	encoder = LabelEncoder()
	X['cuisine'] = encoder.fit_transform(X['cuisine'])
	return X, encoder

def loadTestSet(dir='../data/test.json'):
	"""
	Read in JSON to create test set.
	"""
	import json
	from pandas import DataFrame
	return DataFrame([ExtractRecipe(x).get_predict() for x in json.load(open(dir,'rb'))])

def fitSklearn(X,y,cv,i,model,multi=False):
	"""
	Train a sklearn pipeline or model -- wrapper to enable parallel CV.
	"""
	tr = cv[i][0]
	vl = cv[i][1]
	model.fit(X.iloc[tr],y.iloc[tr])
	if multi:
		return  {"pred": model.predict_proba(X.iloc[vl]), "index":vl}
	else:
		return  {"pred": model.predict_proba(X.iloc[vl])[:,1], "index":vl}

def trainSklearn(model,grid,train,target,cv,refit=True,n_jobs=5,multi=False):
	"""
	Train a sklearn pipeline or model using textual data as input.
	"""
	from joblib import Parallel, delayed   
	from sklearn.grid_search import ParameterGrid
	from numpy import zeros
	if multi:
		pred = zeros((train.shape[0],target.unique().shape[0]))
		from sklearn.metrics import accuracy_score
		score_func = accuracy_score
	else:
		from sklearn.metrics import roc_auc_score
		score_func = roc_auc_score
		pred = zeros(train.shape[0])
	best_score = 0
	for g in ParameterGrid(grid):
		model.set_params(**g)
		if len([True for x in g.keys() if x.find('nthread') != -1 ]) > 0:
			results = [fitSklearn(train,target,list(cv),i,model,multi) for i in range(cv.n_folds)]
		else:
			results = Parallel(n_jobs=n_jobs)(delayed(fitSklearn)(train,target,list(cv),i,model,multi) for i in range(cv.n_folds))
		if multi:
			for i in results:
				pred[i['index'],:] = i['pred']
			score = score_func(target,pred.argmax(1))
		else:
			for i in results:
				pred[i['index']] = i['pred']
			score = score_func(target,pred)
		if score > best_score:
			best_score = score
			best_pred = pred.copy()
			best_grid = g
	print "Best Score: %0.5f" % best_score 
	print "Best Grid", best_grid
	if refit:
		model.set_params(**best_grid)
		model.fit(train,target)
	return best_pred, model

def trainText(model_a,modelGrid_a,model_b,modelGrid_b,train,target,cv,refit=True,n_jobs=5):
	"""
	Train and blend two univariate text models.
	"""
	from sklearn.metrics import roc_auc_score
	from copy import deepcopy
	pred_a, model_a = trainSklearn(deepcopy(model_a),modelGrid_a,train,target,cv,refit=refit,n_jobs=n_jobs)
	pred_b, model_b = trainSklearn(deepcopy(model_b),modelGrid_b,train,target,cv,refit=refit,n_jobs=n_jobs)
	models = TextModel(model_a,model_b)
	best_score = 0
	for w in zip(range(2,100,2),range(98,0,-2)):
		score = roc_auc_score(target,pred_a*w[0]/100.0+pred_b*w[1]/100.0)
		if score > best_score:
			best_score = score
			models.set_weights(w[0]/100.0,w[1]/100.0)
	final_pred = models.blend(pred_a,pred_b)
	print "A Weight:",models.a_weight
	print "B Weight:", models.b_weight
	print "Best Blended Score: %0.5f" % roc_auc_score(target,final_pred)
	return final_pred, models

def splitIngredients(X):
	from pandas import Series
	X2 = X.ingredients.str.split(',? ').apply(lambda x: Series(x)).stack().reset_index(level=1, drop=True)
	X2.name = 'ingredient'
	return X[['cuisine']].join(X2)

def fitIngredients(X,cv,i,model):
	"""
	Train a sklearn pipeline or model -- wrapper to enable parallel CV.
	"""
	from operator import add
	from pandas import Series
	tr = cv[i][0]
	vl = cv[i][1]
	X2 = splitIngredients(X.iloc[tr])
	model.fit(X2.ingredient,X2.cuisine)
	return  {"pred":X.iloc[vl].ingredients.str.split(',? ').apply(lambda recipe:  Series(reduce(add,[model.predict_proba([x]) for x in recipe])[0]/len(recipe))), "index":vl}

def trainIngredient(model,grid,train,cv,refit=True,n_jobs=5):
	from joblib import Parallel, delayed   
	from sklearn.grid_search import ParameterGrid
	from numpy import zeros
	from sklearn.metrics import accuracy_score
	pred = zeros((train.shape[0],train.cuisine.unique().shape[0]))
	best_score = 0
	for g in ParameterGrid(grid):
		model.set_params(**g)
		results = Parallel(n_jobs=n_jobs)(delayed(fitIngredients)(train,list(cv),i,model) for i in range(cv.n_folds))
		for i in results:
			pred[i['index'],:] = i['pred']
		score = accuracy_score(train.cuisine,pred.argmax(1))
		if score > best_score:
			best_score = score
			best_pred = pred.copy()
			best_grid = g
	print "Best Score: %0.5f" % best_score 
	print "Best Grid", best_grid
	if refit:
		X2 = splitIngredients(train)
		model.set_params(**best_grid)
		model.fit(X2.ingredient,X2.cuisine)
	return best_pred, IngredientModel(model)
	
def trainFeatureModel(train,target,model,grid,cv,n_jobs=-1):
	from sklearn.grid_search import ParameterGrid
	from sklearn.metrics import accuracy_score
	from joblib import Parallel, delayed  
	from numpy import zeros
	pred = zeros((train.shape[0],target.unique().shape[0]))
	best_score = 0
	best_grid = {}
	for g in ParameterGrid(grid):
		model.set_params(**g)
		if len([True for x in g.keys() if x.find('nthread') != -1 or x.find('n_jobs') != -1 ]) > 0:
			results = [fitSklearn(train,target,list(cv),i,model,True) for i in range(cv.n_folds)]
		else:
			results = Parallel(n_jobs=n_jobs)(delayed(fitSklearn)(train,target,list(cv),i,model,True) for i in range(cv.n_folds))
		for i in results:
			pred[i['index'],:] = i['pred']
		score = accuracy_score(target,pred.argmax(1))
		if score > best_score:
			best_score = score
			best_pred = pred.copy()
			best_grid = g
	print "Best Score: %0.5f" % best_score 
	print "Best Grid:", best_grid
	model.set_params(**best_grid)
	model.fit(train,target)
	return best_pred, model

    

