"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{function_type = () -> (tensor<i32>, tensor<f32>, tensor<i1>, tensor<4x4xf32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}, {jax.result_info = "[2]", mhlo.layout_mode = "default"}, {jax.result_info = "[3]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
    %1 = "stablehlo.constant"() <{value = dense<4> : tensor<i32>}> : () -> tensor<i32>
    %2 = "stablehlo.constant"() <{value = dense<5.300000e+00> : tensor<f32>}> : () -> tensor<f32>
    %3 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    "func.return"(%1, %2, %3, %0) : (tensor<i32>, tensor<f32>, tensor<i1>, tensor<4x4xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

