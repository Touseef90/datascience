from sklearn import tree, svm
from sklearn.linear_model import SGDClassifier

#  [height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47], [175,64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

clf_d = tree.DecisionTreeClassifier()
clf_d = clf_d.fit(X,Y)
prediction_d = clf_d.predict([[190,70,43]])

clf_s = SGDClassifier(loss="hinge", penalty='l2', tol=None,  max_iter=5)
clf_s = clf_s.fit(X,Y)
prediction_s = clf_s.predict([[190,70,43]])

clf_v = svm.SVC()
clf_v = clf_v.fit(X,Y)
prediction_v = clf_v.predict([[190,70,43]])

print(prediction_d[0]+' - Prediction of DecisionTreeClassifier')
print(prediction_s[0]+' - Prediction of SGDClassifier')
print(prediction_v[0]+' - Prediction of SVC')
