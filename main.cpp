#include <iostream>
#include <vector>
#include <random>
#include <stdlib.h>

using namespace std;
//using vector

vector<float> inputSet {
    0, 0, 0, 1,
    0, 1, 0, 0,
    0, 0, 0, 0,
    1, 1, 0, 1
};

vector<float> rOutPut {
    1, 1, 0, 1
};

vector<float> iWeights {
    0.5,
    0.5,
    0.5,
    0.5
};

vector<float> X {
    5.1, 3.5, 1.4, 0.2,
    4.9, 3.0, 1.4, 0.2,
    6.2, 3.4, 5.4, 2.3,
    5.9, 3.0, 5.1, 1.8
};

vector<float> y {
    0,
    0,
    1,
    1 };

vector<float> W {
    0.5,
    0.5,
    0.5,
    0.5};

vector <float> sigmoid(const vector <float> &m1){
    /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);

    for(unsigned i = 0; i != VECTOR_SIZE; ++i){
        output[i] = 1 / (1 + exp(-m1[i]));
    }

    return output;
}

vector <float> sigmoid_d(const vector <float> &m1){
    //returns the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
    //of the values in vector m1

    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);

    for(unsigned i = 0; i != VECTOR_SIZE; ++i){
        output[i] = m1[i] * (1 - m1[i]);
    }

    return output;
}

vector <float> vecAdd(const vector <float> &m1, const vector <float> &m2){
    //returns the elementwise sum of two vectors

    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);

    for(unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum [i] = m1[i] + m2[i];
    }

    return sum;
}

vector <float> vecSub(const vector <float> &m1, const vector <float> &m2){
    //returns the difference between two vectors (elementwise)

    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);

    for(unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference [i] = m1[i] - m2[i];
    }

    return difference;
}

vector <float> vecMult(const vector <float> &m1, const vector <float> &m2){
    //returns the product of two vectors (elementwise)

    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);

    for(unsigned i = 0; i != VECTOR_SIZE; ++i){
        product [i] = m1[i] * m2[i];
    }

    return product;
}

vector <float> transpose (float *m, const int C, const int R) {

    //Returns a transpose matrix of input matrix.

    vector <float> mT (C*R);

    for(unsigned n = 0; n != C*R; n++) {
        unsigned i = n/C;
        unsigned j = n%C;
        mT[n] = m[R*j + i];
    }

    return mT;
}

vector <float> dot(const vector<float> &m1, const vector<float> &m2, const int m1Rows, const int m1Cols, const int m2Cols){

    //returns the product of two matrices m1 x m2

    vector <float> output (m1Rows * m2Cols);

    for(int row = 0; row != m1Rows; ++row){
        for(int col = 0; col != m2Cols; ++col){
            output[row * m2Cols + col] = 0.f;
            for(int k = 0; k != m1Cols; ++k){
                output[ row * m2Cols + col ] += m1[ row * m1Cols + k ] * m2[ k * m2Cols + col ];
            }
        }
    }

    return output;
}

void print ( const vector <float>& m, int n_rows, int n_columns ) {

    /*  "Couts" the input vector as n_rows x n_columns matrix.
        Inputs:
            m: vector, matrix of size n_rows x n_columns
            n_rows: int, number of rows in the left matrix m1
            n_columns: int, number of columns in the left matrix m1
    */

    for( int i = 0; i != n_rows; ++i ) {
        for( int j = 0; j != n_columns; ++j ) {
            cout << m[ i * n_columns + j ] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}


int main()
{

    for(unsigned i = 0; i != 100000; ++i){
        vector<float> pred = sigmoid(dot(inputSet, W, 4, 4, 1));
        vector<float> pred_error = vecSub(rOutPut, pred);
        vector<float> pred_delta = vecMult(pred_error, sigmoid_d(pred));
        vector<float> W_delta = dot(transpose(&inputSet[0], 4, 4), pred_delta, 4, 4, 1);
        W = vecAdd(W, W_delta);

        if (i == 99999){
            print ( pred, 4, 1);
        }
    }


    return 0;
}
