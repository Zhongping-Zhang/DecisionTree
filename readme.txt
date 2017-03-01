This project judge different types of iris flower by implementing a decision tree.
The 'iris.data' can be downloaded from 'http://www.ics.uci.edu/~mlearn/MLRepository.html'

The 'Decision_Tree' folder includes 'main.py','DecisionTree.py','treePlotter.py' and 'iris.data'.

Function list of 'DecisionTree.py':
1）array_convert_to_list(dataSet)：		convert 'np.array' type to 'list' type
2）Calculate_Shannon_Entropy(dataSet)：		Calculate Shannon Entropy of the dataset
3）Gini_Index(dataSet)：			Calculate Gini index of the dataset
4）Misclassification_Error(dataSet)：		Calculate Misclassification error of the dataset
5）Classify_Dataset(dataSet,position,value)：	extract the sub-dataset of one specific category
6）chooseBestFeatureToSplit(dataSet,method)：	choose one of the best feature to classify the dataset,
	'method' represents three different kinds of decision classification methods, which can be one  
	of 'Entropy','Gini Index' and 'Misclassification error'
7）majorityFeatureClassification(classList)：	choose the majority feature of a list
8）createTree(dataSet,labels,method,num_attributes,depth_tree)：
	Implement decision tree. 
	'dataSet' represents the input dataset. 	
	'labels' represents the attributes of the input dataset.
	'method' is identical to the 'method' in 'chooseBestFeatureToSplit' function.
	'num_attributes' means the total number of attributes of the dataset.
	'depth_tree' is the depth of dicision tree you can input. 
9）store(data)：				Process the data of 'iris.data'
10）createDataSet()：				Create a test dataset.
11）Iris_DataSet()：				Imread 'iris.data', and convert it to a 'list' data.

'main.py' file:
Classify iris dataset by three different division standards(Entropy/Gini Index/Misclassification error).
Can input various depth of tree to watch the effect and training accurancy of different decision tree.


这个项目使用决策树对花朵属性进行分类，从而判断花朵类型。
数据可以从'http://www.ics.uci.edu/~mlearn/MLRepository.html'上下载。

主执行函数：main() 函数
决策树运算函数：DecisionTree() 函数
决策树画图函数：treePlotter() 函数（画图函数可以在“机器学习实战”这本书第三章找到）

DecisionTree()函数 子函数列表：
1）array_convert_to_list(dataSet)：		矩阵类型变量转列表类型变量
2）Calculate_Shannon_Entropy(dataSet)：		计算数据集的香农熵
3）Gini_Index(dataSet)：			计算数据集的基尼指数
4）Misclassification_Error(dataSet)：		计算数据集的误分类指数
5）Classify_Dataset(dataSet,position,value)：	提取数据集某一类的子数据集
6）chooseBestFeatureToSplit(dataSet,method)：	选择分类效果最优的一种属性，method代表分类效果指标，
	分别可以输入'Entropy','Gini Index','Misclassification error'这三种方式。
7）majorityFeatureClassification(classList)：	选择出一个列表中元素数量最多的一种特征
8）createTree(dataSet,labels,method,num_attributes,depth_tree)：
	执行决策树。dataSet代表数据集，labels代表数据集的属性标签，method代表不同的分类方式，
	同chooseBestFeatureToSplit函数的method，num_attributes代表数据集的属性总数，depth_tree
	代表选择决策树的深度。
9）store(data)：				分隔data，用于对读取的iris.data数据进行处理
10）createDataSet()：				创建测试数据集
11）Iris_DataSet()：				读入'iris.data'，并转化为createTree函数可识别的列表形式


main()函数：
分别使用三种不同的标准（Entropy/Gini Index/Misclassification error）进行决策树分类
可以输入不同的决策树深度来观察效果













