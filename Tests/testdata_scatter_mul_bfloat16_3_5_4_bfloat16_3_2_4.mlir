"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<3x5x4xbf16>, tensor<2x1xi64>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> ()
    "func.return"(%6) : (tensor<3x5x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[2.656250e+00, 4.875000e+00, -1.234380e+00, -2.125000e+00], [-2.687500e+00, -2.226560e-01, -2.203130e+00, -1.101560e+00], [3.890630e+00, 1.092530e-02, -2.234380e+00, 2.687500e+00], [-1.923830e-01, 3.093750e+00, 2.234380e+00, -2.890630e-01], [-8.496090e-02, -4.781250e+00, -2.187500e+00, -3.437500e+00]], [[-2.156250e+00, 3.750000e+00, 6.781250e+00, 4.843750e+00], [-3.703130e+00, 1.898440e+00, -6.531250e+00, -2.781250e+00], [3.671880e+00, 1.578130e+00, -1.718750e+00, 1.078130e+00], [3.906250e-01, -3.359380e+00, 7.656250e-01, 2.875000e+00], [-4.980470e-01, -8.007810e-01, 9.414060e-01, 2.796880e+00]], [[-4.406250e+00, 8.740230e-02, -4.125000e+00, -2.671880e+00], [1.664060e+00, 3.140630e+00, -2.765630e+00, -5.562500e+00], [-2.656250e+00, -8.593750e-01, -1.476560e+00, 4.492190e-01], [5.375000e+00, 1.507810e+00, -3.500000e+00, 1.898440e+00], [-2.328130e+00, 2.490230e-01, -9.101560e-01, -9.921870e-01]]]> : tensor<3x5x4xbf16>}> : () -> tensor<3x5x4xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-3.164060e-01, -3.750000e+00, 8.515620e-01, 3.843750e+00], [-2.031250e+00, 6.015630e-01, -1.531250e+00, -8.544920e-02]], [[4.718750e+00, -4.468750e+00, -9.062500e-01, 4.453130e-01], [6.500000e+00, -3.964840e-01, -8.632810e-01, 2.234380e+00]], [[3.843750e+00, -8.875000e+00, -1.757810e+00, -6.406250e+00], [-1.984380e+00, -7.304680e-01, -4.218750e-01, -3.453130e+00]]]> : tensor<3x2x4xbf16>}> : () -> tensor<3x2x4xbf16>
    "func.return"(%1, %2) : (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[2.656250e+00, 4.875000e+00, -1.234380e+00, -2.125000e+00], [-1.726560e+00, 5.039060e-01, 2.875000e+00, 3.613280e-01], [3.890630e+00, 1.092530e-02, -2.234380e+00, 2.687500e+00], [-1.923830e-01, 3.093750e+00, 2.234380e+00, -2.890630e-01], [-8.496090e-02, -4.781250e+00, -2.187500e+00, -3.437500e+00]], [[-2.156250e+00, 3.750000e+00, 6.781250e+00, 4.843750e+00], [-1.140000e+02, 3.375000e+00, -5.093750e+00, -2.781250e+00], [3.671880e+00, 1.578130e+00, -1.718750e+00, 1.078130e+00], [3.906250e-01, -3.359380e+00, 7.656250e-01, 2.875000e+00], [-4.980470e-01, -8.007810e-01, 9.414060e-01, 2.796880e+00]], [[-4.406250e+00, 8.740230e-02, -4.125000e+00, -2.671880e+00], [-1.268750e+01, 2.037500e+01, -2.062500e+00, -1.235000e+02], [-2.656250e+00, -8.593750e-01, -1.476560e+00, 4.492190e-01], [5.375000e+00, 1.507810e+00, -3.500000e+00, 1.898440e+00], [-2.328130e+00, 2.490230e-01, -9.101560e-01, -9.921870e-01]]]> : tensor<3x5x4xbf16>}> : () -> tensor<3x5x4xbf16>
    "func.return"(%0) : (tensor<3x5x4xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

