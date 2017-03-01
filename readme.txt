This project judge different types of iris flower by implementing a decision tree.
The 'iris.data' can be downloaded from 'http://www.ics.uci.edu/~mlearn/MLRepository.html'

The 'Decision_Tree' folder includes 'main.py','DecisionTree.py','treePlotter.py' and 'iris.data'.

Function list of 'DecisionTree.py':
1��array_convert_to_list(dataSet)��		convert 'np.array' type to 'list' type
2��Calculate_Shannon_Entropy(dataSet)��		Calculate Shannon Entropy of the dataset
3��Gini_Index(dataSet)��			Calculate Gini index of the dataset
4��Misclassification_Error(dataSet)��		Calculate Misclassification error of the dataset
5��Classify_Dataset(dataSet,position,value)��	extract the sub-dataset of one specific category
6��chooseBestFeatureToSplit(dataSet,method)��	choose one of the best feature to classify the dataset,
	'method' represents three different kinds of decision classification methods, which can be one  
	of 'Entropy','Gini Index' and 'Misclassification error'
7��majorityFeatureClassification(classList)��	choose the majority feature of a list
8��createTree(dataSet,labels,method,num_attributes,depth_tree)��
	Implement decision tree. 
	'dataSet' represents the input dataset. 	
	'labels' represents the attributes of the input dataset.
	'method' is identical to the 'method' in 'chooseBestFeatureToSplit' function.
	'num_attributes' means the total number of attributes of the dataset.
	'depth_tree' is the depth of dicision tree you can input. 
9��store(data)��				Process the data of 'iris.data'
10��createDataSet()��				Create a test dataset.
11��Iris_DataSet()��				Imread 'iris.data', and convert it to a 'list' data.

'main.py' file:
Classify iris dataset by three different division standards(Entropy/Gini Index/Misclassification error).
Can input various depth of tree to watch the effect and training accurancy of different decision tree.


�����Ŀʹ�þ������Ի������Խ��з��࣬�Ӷ��жϻ������͡�
���ݿ��Դ�'http://www.ics.uci.edu/~mlearn/MLRepository.html'�����ء�

��ִ�к�����main() ����
���������㺯����DecisionTree() ����
��������ͼ������treePlotter() ��������ͼ���������ڡ�����ѧϰʵս���Ȿ��������ҵ���

DecisionTree()���� �Ӻ����б�
1��array_convert_to_list(dataSet)��		�������ͱ���ת�б����ͱ���
2��Calculate_Shannon_Entropy(dataSet)��		�������ݼ�����ũ��
3��Gini_Index(dataSet)��			�������ݼ��Ļ���ָ��
4��Misclassification_Error(dataSet)��		�������ݼ��������ָ��
5��Classify_Dataset(dataSet,position,value)��	��ȡ���ݼ�ĳһ��������ݼ�
6��chooseBestFeatureToSplit(dataSet,method)��	ѡ�����Ч�����ŵ�һ�����ԣ�method�������Ч��ָ�꣬
	�ֱ��������'Entropy','Gini Index','Misclassification error'�����ַ�ʽ��
7��majorityFeatureClassification(classList)��	ѡ���һ���б���Ԫ����������һ������
8��createTree(dataSet,labels,method,num_attributes,depth_tree)��
	ִ�о�������dataSet�������ݼ���labels�������ݼ������Ա�ǩ��method����ͬ�ķ��෽ʽ��
	ͬchooseBestFeatureToSplit������method��num_attributes�������ݼ�������������depth_tree
	����ѡ�����������ȡ�
9��store(data)��				�ָ�data�����ڶԶ�ȡ��iris.data���ݽ��д���
10��createDataSet()��				�����������ݼ�
11��Iris_DataSet()��				����'iris.data'����ת��ΪcreateTree������ʶ����б���ʽ


main()������
�ֱ�ʹ�����ֲ�ͬ�ı�׼��Entropy/Gini Index/Misclassification error�����о���������
�������벻ͬ�ľ�����������۲�Ч��













