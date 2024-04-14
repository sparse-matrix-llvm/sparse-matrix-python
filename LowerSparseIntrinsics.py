import pprint
import numpy as np

class LowerSparseIntrinsics:
    def __init__(self, dense_matrix=None, num_rows=0, num_cols=0):
        self.shape_info = self.ShapeInfo(num_rows, num_cols)
        self.csc_matrix = self.CSCMatrix()
        if dense_matrix:
            self.dense_to_csc(dense_matrix)
    
    class ShapeInfo:
        def __init__(self, num_rows=0, num_cols=0):
            self.num_rows = num_rows
            self.num_cols = num_cols
    
    class CSCMatrix:
        def __init__(self):
            # these would all be small vectors just like in normal matrix
            self.values = []
            self.row_indices = []
            self.col_pointers = [0]
        
        def __repr__(self):
            return f"csc_matrix(values={self.values}, row_indices={self.row_indices}, col_pointers{self.col_pointers})"

        def __eq__(self, other):
            return self.values == other.values and self.row_indices == other.row_indices and self.col_pointers == other.col_pointers
        
    
    def dense_to_csc(self, dense_matrix):
        for column in dense_matrix:
            column_nonzeros = 0
            for row_index, value in enumerate(column):
                if value != 0:
                    self.csc_matrix.values.append(value)
                    self.csc_matrix.row_indices.append(row_index)
                    column_nonzeros += 1
            self.csc_matrix.col_pointers.append(self.csc_matrix.col_pointers[-1] + column_nonzeros)

    def csc_to_column_vectors(csc_matrix):
        num_cols = csc_matrix.shape_info.num_cols
        num_rows = csc_matrix.shape_info.num_rows
        column_vectors = [[] for _ in range(num_cols)]
    
        # Process each column defined by col_pointers
        for col in range(num_cols):
            start_index = csc_matrix.csc_matrix.col_pointers[col]
            end_index = csc_matrix.csc_matrix.col_pointers[col + 1]

            # Create a temporary column with all zeros
            column = [0] * num_rows

            # Fill in the non-zero values from the CSC data
            for i in range(start_index, end_index):
                row = csc_matrix.csc_matrix.row_indices[i]
                value = csc_matrix.csc_matrix.values[i]
                column[row] = value

            # Append the populated column to the list of column vectors
            column_vectors[col] = column

        return column_vectors


    def csc_to_flat_array(self):
        # we can get len(self.col_pointers) and len(self.values) from SmallVectors.size()
        flat_array = [self.csc_matrix.col_pointers[-1]] + self.csc_matrix.col_pointers + self.csc_matrix.row_indices + self.csc_matrix.values
        return flat_array
    
    def flat_array_to_csc(self,flat_array, num_rows, num_cols):
        nnz = flat_array[0]
        offset_col_pointers = 1
        offset_row_indices = offset_col_pointers + num_cols + 1
        offset_values = offset_row_indices + nnz
    
        col_pointers = flat_array[offset_col_pointers:offset_row_indices]
        row_indices = flat_array[offset_row_indices:offset_values]
        values = flat_array[offset_values:offset_values + nnz]
        
        csc_matrix = self.CSCMatrix()
        csc_matrix.col_pointers = col_pointers
        csc_matrix.row_indices = row_indices
        csc_matrix.values = values
        
        return csc_matrix

    @staticmethod
    # # col by col multiply from: "A Systematic Survey of General Sparse Matrix-Matrix Multiplication" section 3.5
    def csc_matrix_multiplication(sparse_a, sparse_b):
        A_values = sparse_a.csc_matrix.values
        A_row_indices = sparse_a.csc_matrix.row_indices
        A_col_pointers = sparse_a.csc_matrix.col_pointers

        B_values = sparse_b.csc_matrix.values
        B_row_indices = sparse_b.csc_matrix.row_indices
        B_col_pointers = sparse_b.csc_matrix.col_pointers

        C_values = []
        C_row_indices = []
        C_col_pointers = [0]  # The first column pointer is always 0

        for j in range(sparse_b.shape_info.num_cols):
            # Dictionary to hold the non-zero values of column j of C
            column_values = {}

            # Multiply the non-zero entries of column j in B by the corresponding columns in A
            for k_index in range(B_col_pointers[j], B_col_pointers[j + 1]):
                k = B_row_indices[k_index]
                bkj = B_values[k_index]
                for i_index in range(A_col_pointers[k], A_col_pointers[k + 1]):
                    i = A_row_indices[i_index]
                    aik = A_values[i_index]
                    column_values[i] = column_values.get(i, 0) + aik * bkj

            # Sort the row indices and append the values and row indices to the C arrays
            for i in sorted(column_values):
                C_values.append(column_values[i])
                C_row_indices.append(i)

            # Update the column pointers for each column processed
            C_col_pointers.append(len(C_values))

        # Construct the resulting CSC matrix
        c_sparse = LowerSparseIntrinsics()
        c_sparse.csc_matrix.values = C_values
        c_sparse.csc_matrix.row_indices = C_row_indices
        c_sparse.csc_matrix.col_pointers = C_col_pointers
        c_sparse.shape_info.num_rows = sparse_a.shape_info.num_rows
        c_sparse.shape_info.num_cols = sparse_b.shape_info.num_cols

        return c_sparse

    
    
if __name__ == "__main__":
    """
    row representation
    [1, 0, 0, 0],
    [0, 2, 0, 3],
    [4, 0, 5, 0],
    [0, 0, 0, 6]
    """
    dense_matrix = [
    [1, 0, 4, 0],  # Column 1
    [0, 2, 0, 0],  # Column 2
    [0, 0, 5, 0],  # Column 3
    [0, 3, 0, 6]   # Column 4
    ]

    csc_matrix = LowerSparseIntrinsics(dense_matrix, len(dense_matrix), len(dense_matrix[0]))
    flat_array = csc_matrix.csc_to_flat_array()
    new_csc_matrix = csc_matrix.flat_array_to_csc(flat_array, len(dense_matrix), len(dense_matrix[0]))

    assert new_csc_matrix == csc_matrix.csc_matrix

    csc_matrix_2 = LowerSparseIntrinsics(dense_matrix, len(dense_matrix), len(dense_matrix[0]))
    res = csc_matrix.csc_matrix_multiplication(csc_matrix, csc_matrix_2)
    print(res.csc_matrix)
    res_dense = res.csc_to_column_vectors()

    print(res_dense)

    dense_a = np.array(dense_matrix).T
    dense_b = np.array(dense_matrix).T
    print((dense_a @ dense_b).T)
