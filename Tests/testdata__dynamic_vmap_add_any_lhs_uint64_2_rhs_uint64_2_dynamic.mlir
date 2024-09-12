"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2xui64>, tensor<?x2xui64>) -> tensor<?x2xui64>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2xui64>, %arg2: tensor<?x2xui64>):
    %0 = "stablehlo.add"(%arg1, %arg2) : (tensor<?x2xui64>, tensor<?x2xui64>) -> tensor<?x2xui64>
    "func.return"(%0) : (tensor<?x2xui64>) -> ()
  }) : () -> ()
}) : () -> ()

