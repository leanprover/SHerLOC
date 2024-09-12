"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xf64>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xf64>, tensor<3x4xf64>) -> ()
    "func.return"(%2) : (tensor<3x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.93871766744870521, 2.8357154514833556, -1.9676780707688346, 3.0538035765256391], [1.2355984062438199, -4.6736242515759301, 2.8795909205256449, 2.1062487351838799], [1.4274756889977089, 2.6006261663243797, 6.7800422809940102, 4.5164546160017318]]> : tensor<3x4xf64>}> : () -> tensor<3x4xf64>
    "func.return"(%1) : (tensor<3x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0.93871766744870521, 2.8357154514833556, -1.9676780707688346, 3.0538035765256391], [1.2355984062438199, -4.6736242515759301, 2.8795909205256449, 2.1062487351838799], [1.4274756889977089, 2.6006261663243797, 6.7800422809940102, 4.5164546160017318]]> : tensor<3x4xf64>}> : () -> tensor<3x4xf64>
    "func.return"(%0) : (tensor<3x4xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

