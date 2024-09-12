"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<5x2xbf16>, tensor<5x2xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<5x3xbf16>
    %4:2 = "func.call"() <{callee = @expected}> : () -> (tensor<5x2xbf16>, tensor<5x2xi32>)
    %5:2 = "chlo.top_k"(%3) <{k = 2 : i64}> : (tensor<5x3xbf16>) -> (tensor<5x2xbf16>, tensor<5x2xi32>)
    "stablehlo.custom_call"(%5#0, %4#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xbf16>, tensor<5x2xbf16>) -> ()
    "stablehlo.custom_call"(%5#1, %4#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    "func.return"(%5#0, %5#1) : (tensor<5x2xbf16>, tensor<5x2xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[5.062500e+00, 2.781250e+00, -2.093750e+00], [-2.968750e+00, 7.695310e-01, 3.203130e+00], [1.679690e+00, 1.601560e-01, -3.468750e+00], [-1.359380e+00, 3.765630e+00, -9.179680e-01], [1.242190e+00, 3.484380e+00, 4.277340e-01]]> : tensor<5x3xbf16>}> : () -> tensor<5x3xbf16>
    "func.return"(%2) : (tensor<5x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x2xbf16>, tensor<5x2xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[5.062500e+00, 2.781250e+00], [3.203130e+00, 7.695310e-01], [1.679690e+00, 1.601560e-01], [3.765630e+00, -9.179680e-01], [3.484380e+00, 1.242190e+00]]> : tensor<5x2xbf16>}> : () -> tensor<5x2xbf16>
    %1 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 1], [0, 1], [1, 2], [1, 0]]> : tensor<5x2xi32>}> : () -> tensor<5x2xi32>
    "func.return"(%0, %1) : (tensor<5x2xbf16>, tensor<5x2xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

