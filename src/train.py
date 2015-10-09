#!/na/home/dmcgarry/envs/ml/bin/python
"""
Trains the models for Kaggle's "What's Cooking" contest: https://www.kaggle.com/c/whats-cooking

__author__ = "David McGarry"
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import cPickle as pickle
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from helper import *

#############
### Grids ###
#############

ingred_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents='unicode',analyzer="char",preprocessor=stripString)),
    ('feat', SelectPercentile(chi2)),
    ('model', LogisticRegression())
])
ingred_grid = {
    'tfidf__ngram_range':[(2,6)],
    'feat__percentile':[95,90,85],
    'model__C':[5]
}

pipe_glm = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents='unicode',
    	analyzer="char",preprocessor=stripString)),
    ('feat', SelectPercentile(chi2)),
    ('model', LogisticRegression())
])
grid_glm = {
	'greek':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [37]
	},
	'southern_us':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [57]
	},
	'filipino':{
		'model__C': [7], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [65]
	},
	'indian':{
		'model__C': [3], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [52]
	},
	'jamaican':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [99]
	},
	'spanish':{
		'model__C': [3], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [88]
	},
	'italian':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [91]
	},
	'mexican':{
		'model__C': [7], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [93]
	},
	'chinese':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [76]
	},
	'british':{
		'model__C': [10], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [75]
	},
	'thai':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [97]
	},
	'vietnamese':{
		'model__C': [3], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [27]
	},
	'cajun_creole':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [88]
	},
	'brazilian':{
		'model__C': [10], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [98]
	},
	'french':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [46]
	},
	'japanese':{
		'model__C': [5], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [97]
	},
	'irish':{
		'model__C': [8], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [17,15,13]
	},
	'korean':{
		'model__C': [8], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [54]
	},
	'moroccan':{
		'model__C': [10], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [55]
	},
	'russian':{
		'model__C': [6], 
		'tfidf__ngram_range': [(2, 6)], 
		'feat__percentile': [91]
	}
}

pipe_xgb = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents='unicode',
    	tokenizer=LemmaTokenizer())),
    ('model', xgb.XGBClassifier())
])
grid_xgb = {
	'greek':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.08], 
		'model__max_depth': [6], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'southern_us':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.10], 
		'model__max_depth': [16], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'filipino':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.08], 
		'model__max_depth': [4], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'indian':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.06], 
		'model__max_depth': [8], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'jamaican':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.04], 
		'model__max_depth': [14], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'spanish':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.08], 
		'model__max_depth': [14], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'italian':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.14], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'mexican':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.12], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'chinese':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.10], 
		'model__max_depth': [6], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'british':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.06], 
		'model__max_depth': [10], 
		'model__n_estimators': [400], 
		'model__subsample': [0.6]
	},
	'thai':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.14], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'vietnamese':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.04], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'cajun_creole':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.06], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'brazilian':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.04], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'french':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.10], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'japanese':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.08], 
		'model__max_depth': [8], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'irish':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.06], 
		'model__max_depth': [14], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'korean':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.10], 
		'model__max_depth': [6], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'moroccan':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.10], 
		'model__max_depth': [8], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	},
	'russian':{
		'tfidf__ngram_range': [(1, 2)], 
		'model__nthread': [20], 
		'model__learning_rate': [0.06], 
		'model__max_depth': [10], 
		'model__n_estimators': [500], 
		'model__subsample': [0.6]
	}
}


vars = [u'ingredient_count',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
        u'pred_text_greek', u'pred_text_southern_us', u'pred_text_filipino',
    	u'pred_text_indian', u'pred_text_jamaican', u'pred_text_spanish', 
    	u'pred_text_italian', u'pred_text_mexican', u'pred_text_chinese', 
    	u'pred_text_british', u'pred_text_thai', u'pred_text_vietnamese', 
    	u'pred_text_cajun_creole', u'pred_text_brazilian', u'pred_text_french', 
    	u'pred_text_japanese', u'pred_text_irish', u'pred_text_korean', 
    	u'pred_text_moroccan', u'pred_text_russian']


pipe_xgb_final = Pipeline([
    ('union', FeatureUnion([
        ('lsa', Pipeline([
            ('var', VarSelect(keys='ingredients')),
        	('tfidf', TfidfVectorizer(
        		strip_accents='unicode',analyzer="char",
        		ngram_range=(2,6),preprocessor=stripString)),
            ('svd', TruncatedSVD())
        ])),
        ('feat', Pipeline([
            ('var', VarSelect(keys=vars))
        ]))
    ])),
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
    ('feat', SelectKBest(f_classif)),
    ('model',xgb.XGBClassifier())
])
grid_xgb_final = {
	'union__lsa__svd__n_components':[50],
    'feat__k':[85,75],
    'model__n_estimators':[750],
    'model__learning_rate': [0.08],
    'model__max_depth':[16,18],
    'model__subsample': [0.65],
    'model__nthread':[20]
}

pipe_rf_final = Pipeline([
    ('union', FeatureUnion([
        ('lsa', Pipeline([
            ('var', VarSelect(keys='ingredients')),
        	('tfidf', TfidfVectorizer(
        		strip_accents='unicode',analyzer="char",
        		ngram_range=(2,6),tokenizer=LemmaTokenizer())),
            ('svd', TruncatedSVD())
        ])),
        ('feat', Pipeline([
            ('var', VarSelect(keys=vars))
        ]))
    ])),
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
    ('feat', SelectKBest(f_classif)),
    ('model', RandomForestClassifier())
])
grid_rf_final = {
	'union__lsa__svd__n_components':[60,75,90],
    'feat__k':[80],
    'model__n_estimators':[250,500,750],
    'model__max_features':[8],
    'model__max_depth':[35],
    'model__n_jobs':[20]
}

############
### Main ###
############

def main():
	# load data
	train, encoder = loadTrainSet()
	cv = KFold(train.shape[0], n_folds=8, shuffle=True)

	# train ingredient model
	print "Ingredient Model"
	ingred_pred, ingred_model = trainIngredient(ingred_pipe,ingred_grid,train,cv,n_jobs=-1)
	train = train.join(pd.DataFrame(ingred_pred))

	# train text models
	text_models = {}
	for v in [encoder.inverse_transform(v) for v in train.cuisine.unique()]:
		print "\nText Model: %s" % v
		train['pred_text_'+v], text_models[v] = trainText(pipe_glm,grid_glm[v],pipe_xgb,grid_xgb[v],train.ingredients,train.cuisine.apply(lambda x: 1 if x == encoder.transform(v) else 0),cv,n_jobs=-1)

	# train feature models
	print "\nFeature Model: xgb"
	xgb_pred, recipe_model_xgb = trainFeatureModel(train,train.cuisine,pipe_xgb_final,grid_xgb_final,cv)
	print "\nFeature Model: rf"
	rf_pred, recipe_model_rf = trainFeatureModel(train,train.cuisine,pipe_rf_final,grid_rf_final,cv)

	# blend feature models into final recipe model
	final_model = RecipeModel(ingred_model,text_models,recipe_model_xgb,recipe_model_rf,encoder)
	final_model.set_weights(xgb_pred,rf_pred,train.cuisine)
	print final_model

	#make predictions
	test = loadTestSet()
	test['cuisine'] = final_model.predict_kaggle(test)
	test[['id','cuisine']].to_csv("../data/pred.csv",index=False)



if __name__ == "__main__":
    main()
