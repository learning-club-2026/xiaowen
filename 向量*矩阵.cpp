#include <iostream>
#include <vector>
using namespace std;

vector<double> vecMultMatrix(vector<double> vec, vector<vector<double>> mat) {
    if (vec.size() != mat.size()) {
        cout << "矩阵行数应与向量长度相等" << endl;
        return {};
    }

    int cols = mat[0].size();
    vector<double> res(cols, 0.0);

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < vec.size(); j++) {
            res[i] += vec[j] * mat[j][i];
        }
    }
    return res;
}

int main() {
    vector<double> vec;
    double num;
    cout << "输入向量元素：";
    while (cin.peek() != '\n') {
        cin >> num;
        vec.push_back(num);
    }

    int row, col;
    cout << "输入矩阵行数和列数：";
    cin >> row >> col;

    vector<vector<double>> mat(row, vector<double>(col));
    cout << "输入矩阵数据：" << endl;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cin >> mat[i][j];
        }
    }

    vector<double> result = vecMultMatrix(vec, mat);
    cout << "结果：";
    for (double x : result) {
        cout << x << " ";
    }
    return 0;
}
