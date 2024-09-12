"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xf32>
    %4 = "stablehlo.reduce_precision"(%2) <{exponent_bits = 11 : i32, mantissa_bits = 52 : i32}> : (tensor<5x7xf32>) -> tensor<5x7xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x7xf32>, tensor<5x7xf32>) -> ()
    "func.return"(%4) : (tensor<5x7xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.0463174656, -0.359622896, 2.68061018, 2.73067665, -2.98465776, 2.37090635, 3.00272846], [-1.70035434, 0.561822534, 2.39265943, -3.71191573, -2.10235906, -3.70896864, 2.49592757], [2.0122726, 0.415704519, -0.829515696, 2.32972074, 1.1138587, 2.00742745, 2.4653914], [5.22061348, 2.39084625, 0.787524521, 0.837473809, 2.02389908, 0.432699025, 1.31516218], [-1.65173602, 2.73440528, 0.9528777, -0.0713662207, 2.9412961, -1.7066927, -3.27265668]]> : tensor<5x7xf32>}> : () -> tensor<5x7xf32>
    "func.return"(%1) : (tensor<5x7xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0.0463174656, -0.359622896, 2.68061018, 2.73067665, -2.98465776, 2.37090635, 3.00272846], [-1.70035434, 0.561822534, 2.39265943, -3.71191573, -2.10235906, -3.70896864, 2.49592757], [2.0122726, 0.415704519, -0.829515696, 2.32972074, 1.1138587, 2.00742745, 2.4653914], [5.22061348, 2.39084625, 0.787524521, 0.837473809, 2.02389908, 0.432699025, 1.31516218], [-1.65173602, 2.73440528, 0.9528777, -0.0713662207, 2.9412961, -1.7066927, -3.27265668]]> : tensor<5x7xf32>}> : () -> tensor<5x7xf32>
    "func.return"(%0) : (tensor<5x7xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

