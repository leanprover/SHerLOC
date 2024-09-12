"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<3x5x4xf16>, tensor<2x1xi64>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> ()
    "func.return"(%6) : (tensor<3x5x4xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-1.936520e+00, 9.487300e-01, -4.304690e+00, -3.754880e-01], [4.511720e+00, 1.542970e+00, -1.715820e+00, -5.146480e-01], [-2.093750e+00, -3.384770e+00, 1.601560e+00, 1.021480e+00], [1.140630e+00, -3.554690e+00, 2.382810e+00, 2.835940e+00], [3.548830e+00, -1.038090e+00, 3.873050e+00, 2.392580e+00]], [[-3.059390e-02, 2.010500e-01, -3.294920e+00, 2.548830e+00], [3.636720e+00, 1.351560e+00, -1.500980e+00, -6.996090e+00], [-6.367190e+00, -2.457030e+00, -3.781250e+00, -2.902830e-01], [1.160160e+00, -3.828130e-01, -3.113280e+00, -4.167970e+00], [-1.818360e+00, -3.687500e+00, 1.850590e-01, -1.429690e+00]], [[-6.007810e+00, -4.602050e-01, -6.083980e-01, 1.167970e+00], [1.500000e+00, 5.640630e+00, -2.316410e+00, 1.067380e+00], [-1.742190e+00, 2.144530e+00, 2.106930e-01, 9.252920e-01], [5.125000e+00, -1.057740e-01, -1.627930e+00, -1.481930e-01], [1.069340e+00, -4.003910e-01, -1.496090e+00, -7.812500e+00]]]> : tensor<3x5x4xf16>}> : () -> tensor<3x5x4xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[2.451170e+00, -9.887690e-01, -3.940430e-01, 2.359380e+00], [-9.501950e-01, 7.879640e-02, 1.696290e+00, -1.478520e+00]], [[1.146480e+00, -1.721680e+00, -7.919920e-01, 3.582030e+00], [2.201170e+00, 3.056640e+00, 8.208000e-01, 2.572270e+00]], [[3.540040e-02, 3.501950e+00, 3.730470e-01, -2.778630e-02], [-5.593750e+00, 3.215330e-01, 6.261710e+00, -2.066410e+00]]]> : tensor<3x2x4xf16>}> : () -> tensor<3x2x4xf16>
    "func.return"(%1, %2) : (tensor<3x5x4xf16>, tensor<3x2x4xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-1.936520e+00, 9.487300e-01, -4.304690e+00, -3.754880e-01], [-1.050780e+01, -1.201780e-01, 1.147460e+00, 1.794920e+00], [-2.093750e+00, -3.384770e+00, 1.601560e+00, 1.021480e+00], [1.140630e+00, -3.554690e+00, 2.382810e+00, 2.835940e+00], [3.548830e+00, -1.038090e+00, 3.873050e+00, 2.392580e+00]], [[-3.059390e-02, 2.010500e-01, -3.294920e+00, 2.548830e+00], [9.171870e+00, -7.109380e+00, 9.755850e-01, -6.443750e+01], [-6.367190e+00, -2.457030e+00, -3.781250e+00, -2.902830e-01], [1.160160e+00, -3.828130e-01, -3.113280e+00, -4.167970e+00], [-1.818360e+00, -3.687500e+00, 1.850590e-01, -1.429690e+00]], [[-6.007810e+00, -4.602050e-01, -6.083980e-01, 1.167970e+00], [-2.971190e-01, 6.351560e+00, -5.410150e+00, 6.130980e-02], [-1.742190e+00, 2.144530e+00, 2.106930e-01, 9.252920e-01], [5.125000e+00, -1.057740e-01, -1.627930e+00, -1.481930e-01], [1.069340e+00, -4.003910e-01, -1.496090e+00, -7.812500e+00]]]> : tensor<3x5x4xf16>}> : () -> tensor<3x5x4xf16>
    "func.return"(%0) : (tensor<3x5x4xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

