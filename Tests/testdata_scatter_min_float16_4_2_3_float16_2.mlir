"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf16>, tensor<2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-4.195310e+00, -2.726560e+00, -2.300780e+00], [5.562500e+00, 5.890630e+00, 1.609380e+00]], [[-1.485350e+00, -1.921880e+00, -3.976560e+00], [2.164060e+00, -4.113280e+00, 1.083980e+00]], [[-1.351560e+00, 5.355470e+00, 7.287590e-02], [-4.378910e+00, 1.294920e+00, -1.518550e+00]], [[3.146480e+00, 3.517580e+00, -1.776120e-01], [-2.851560e+00, 6.285150e+00, -8.886710e-01]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[-2.132810e+00, -8.239740e-02]> : tensor<2xf16>}> : () -> tensor<2xf16>
    "func.return"(%1, %2) : (tensor<4x2x3xf16>, tensor<2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-4.195310e+00, -2.726560e+00, -2.300780e+00], [5.562500e+00, 5.890630e+00, 1.609380e+00]], [[-1.485350e+00, -1.921880e+00, -3.976560e+00], [2.164060e+00, -4.113280e+00, 1.083980e+00]], [[-1.351560e+00, 5.355470e+00, 7.287590e-02], [-4.378910e+00, 1.294920e+00, -1.518550e+00]], [[3.146480e+00, 3.517580e+00, -2.132810e+00], [-2.851560e+00, 6.285150e+00, -8.886710e-01]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    "func.return"(%0) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

