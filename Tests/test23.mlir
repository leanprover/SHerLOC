"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{function_type = () -> (tensor<i32>, tensor<f32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %0 = "stablehlo.constant"() <{value = dense<4> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.constant"() <{value = dense<5.300000e+00> : tensor<f32>}> : () -> tensor<f32>
    "func.return"(%0, %1) : (tensor<i32>, tensor<f32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

