"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<3x5x4xbf16>, tensor<2x1xi64>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> ()
    "func.return"(%6) : (tensor<3x5x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[2.281250e+00, -6.500000e+00, -4.296880e-02, -1.156250e+00], [3.500000e+00, 2.875000e+00, -2.203130e+00, -1.226560e+00], [-1.710940e+00, 1.281250e+00, 1.218750e+00, -4.312500e+00], [-9.414060e-01, 4.902340e-01, 3.109380e+00, 4.812500e+00], [2.734380e+00, -2.871090e-01, -6.093750e-01, -2.484380e+00]], [[-2.203130e+00, 2.353520e-01, 2.921880e+00, 3.390630e+00], [-3.609380e+00, 1.695310e+00, -8.671870e-01, 1.656250e+00], [2.578130e+00, 4.031250e+00, -1.867190e+00, 4.593750e+00], [-2.531250e+00, 5.351560e-01, -2.609380e+00, -1.062500e+00], [-2.328130e+00, 2.437500e+00, 9.218750e-01, -1.789060e+00]], [[7.148430e-01, -1.468750e+00, -2.671880e+00, 2.687500e+00], [1.546880e+00, -3.046880e+00, 5.820310e-01, -8.476560e-01], [1.187500e+00, 2.218750e+00, -7.187500e-01, -4.687500e-01], [-8.515620e-01, -3.875000e+00, -1.953130e+00, 4.375000e+00], [-2.875000e+00, 1.648440e+00, 3.093750e+00, -1.406250e+00]]]> : tensor<3x5x4xbf16>}> : () -> tensor<3x5x4xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[3.218750e+00, 1.718750e+00, 1.031250e+00, -1.578130e+00], [-3.796880e+00, -1.851560e+00, -1.648440e+00, -2.500000e+00]], [[2.609380e+00, 4.257810e-01, 1.953130e+00, -3.609380e+00], [8.500000e+00, -4.468750e+00, 1.429690e+00, 3.015630e+00]], [[-2.625000e+00, 2.281250e+00, -2.015630e+00, -6.289060e-01], [-4.406250e+00, 3.218750e+00, 8.632810e-01, 6.062500e+00]]]> : tensor<3x2x4xbf16>}> : () -> tensor<3x2x4xbf16>
    "func.return"(%1, %2) : (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[2.281250e+00, -6.500000e+00, -4.296880e-02, -1.156250e+00], [2.921880e+00, 2.750000e+00, -2.812500e+00, -5.312500e+00], [-1.710940e+00, 1.281250e+00, 1.218750e+00, -4.312500e+00], [-9.414060e-01, 4.902340e-01, 3.109380e+00, 4.812500e+00], [2.734380e+00, -2.871090e-01, -6.093750e-01, -2.484380e+00]], [[-2.203130e+00, 2.353520e-01, 2.921880e+00, 3.390630e+00], [7.500000e+00, -2.343750e+00, 2.515630e+00, 1.062500e+00], [2.578130e+00, 4.031250e+00, -1.867190e+00, 4.593750e+00], [-2.531250e+00, 5.351560e-01, -2.609380e+00, -1.062500e+00], [-2.328130e+00, 2.437500e+00, 9.218750e-01, -1.789060e+00]], [[7.148430e-01, -1.468750e+00, -2.671880e+00, 2.687500e+00], [-5.500000e+00, 2.453130e+00, -5.742190e-01, 4.593750e+00], [1.187500e+00, 2.218750e+00, -7.187500e-01, -4.687500e-01], [-8.515620e-01, -3.875000e+00, -1.953130e+00, 4.375000e+00], [-2.875000e+00, 1.648440e+00, 3.093750e+00, -1.406250e+00]]]> : tensor<3x5x4xbf16>}> : () -> tensor<3x5x4xbf16>
    "func.return"(%0) : (tensor<3x5x4xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

