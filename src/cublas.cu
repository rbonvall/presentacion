void matrix_vector_product(int nr_rows, int nr_cols,
                           float *b,
                           float *a,
                           float *x) {
    cublasSgemv('t', nr_cols, nr_rows,
      1.0f, a, nr_cols,
            x, 1,
      0.0f, b, 1);
}
