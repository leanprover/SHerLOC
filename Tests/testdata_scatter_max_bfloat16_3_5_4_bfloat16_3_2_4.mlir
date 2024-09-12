"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<3x5x4xbf16>, tensor<2x1xi64>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> ()
    "func.return"(%6) : (tensor<3x5x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-2.921880e+00, -1.101560e+00, -1.796880e+00, 2.328130e+00], [1.953130e+00, -1.289060e+00, -2.015630e+00, 7.382810e-01], [-5.125000e+00, -3.222660e-01, 4.375000e+00, 3.000000e+00], [3.476560e-01, -4.281250e+00, 4.812500e+00, -1.828130e+00], [-2.531250e+00, -7.070310e-01, 3.359380e+00, 5.375000e+00]], [[3.156250e+00, 2.218750e+00, 1.171880e-01, 5.351560e-01], [1.562500e+00, 2.125000e+00, 4.511720e-01, -9.609370e-01], [3.515630e+00, 2.000000e+00, -1.937500e+00, -1.611330e-01], [6.562500e-01, 5.625000e+00, -1.765630e+00, 4.433590e-01], [5.102540e-02, 2.015630e+00, -1.039060e+00, -4.062500e+00]], [[-8.812500e+00, 4.765630e-01, 2.984380e+00, 9.312500e+00], [5.156250e+00, 1.609380e+00, -2.156250e+00, -1.367190e+00], [3.609380e+00, 2.687500e+00, 6.640630e-01, 4.812500e+00], [6.000000e+00, -1.250000e+00, -3.250000e+00, 1.552730e-01], [-1.390630e+00, -5.781250e+00, 8.375000e+00, 1.312500e+00]]]> : tensor<3x5x4xbf16>}> : () -> tensor<3x5x4xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-5.593750e+00, 4.531250e+00, -1.335940e+00, -2.187500e+00], [2.500000e+00, 6.367190e-01, 1.390630e+00, 6.484380e-01]], [[1.921880e+00, -7.937500e+00, -1.132810e+00, 4.609380e-01], [-5.781250e+00, 1.507810e+00, -8.828120e-01, -3.500000e+00]], [[1.656250e+00, -5.156250e+00, 3.218750e+00, -2.312500e+00], [-3.453130e+00, 2.125000e+00, 2.640630e+00, -1.765630e+00]]]> : tensor<3x2x4xbf16>}> : () -> tensor<3x2x4xbf16>
    "func.return"(%1, %2) : (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-2.921880e+00, -1.101560e+00, -1.796880e+00, 2.328130e+00], [2.500000e+00, 4.531250e+00, 1.390630e+00, 7.382810e-01], [-5.125000e+00, -3.222660e-01, 4.375000e+00, 3.000000e+00], [3.476560e-01, -4.281250e+00, 4.812500e+00, -1.828130e+00], [-2.531250e+00, -7.070310e-01, 3.359380e+00, 5.375000e+00]], [[3.156250e+00, 2.218750e+00, 1.171880e-01, 5.351560e-01], [1.921880e+00, 2.125000e+00, 4.511720e-01, 4.609380e-01], [3.515630e+00, 2.000000e+00, -1.937500e+00, -1.611330e-01], [6.562500e-01, 5.625000e+00, -1.765630e+00, 4.433590e-01], [5.102540e-02, 2.015630e+00, -1.039060e+00, -4.062500e+00]], [[-8.812500e+00, 4.765630e-01, 2.984380e+00, 9.312500e+00], [5.156250e+00, 2.125000e+00, 3.218750e+00, -1.367190e+00], [3.609380e+00, 2.687500e+00, 6.640630e-01, 4.812500e+00], [6.000000e+00, -1.250000e+00, -3.250000e+00, 1.552730e-01], [-1.390630e+00, -5.781250e+00, 8.375000e+00, 1.312500e+00]]]> : tensor<3x5x4xbf16>}> : () -> tensor<3x5x4xbf16>
    "func.return"(%0) : (tensor<3x5x4xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

