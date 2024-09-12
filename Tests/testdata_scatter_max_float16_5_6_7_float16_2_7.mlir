"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<2x7xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2xi64>, tensor<2x7xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<2x7xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x213F40B019C44F43ACC41739E72CD3C0DF427740253B67C17FB709421CC14EC4DAB82D4145C71992D6431AC88D2B5CB542B98F37F33DCBBAA9416645E4414234F2C0754558B53935FCC156C2DCB2FEC152B6D84215C6283D26422E2A1D427545AD4081C3BC321E443D3C9FAFE545D9BF26B818B8994308450238ABB2B4C6843E263554B4A3B4EE3C2F3D79C2FF46794401453FC0D5C284C226C058C2D4BA62B71F40E9B7CA457DBD773CE63E3C380831663DDFC0F7C0F0C20FC5873C2F4188BABFBB08C058B9734087A9F9464DC537C6CA3C843E033FFDBC7C3E6EC2E6BD2BC47CC30F3F714175BD014161C2083641442FC11BA8133EAD32B33D53BAD7BE0C312937E2C5DABA2C4173BD01C451ACE8445A402BB676BF01C5BD44FBB6EB364D42453CF5BDBB3D5243FF4135BDCA4832B6DA40B1B4F03B48BE914170C21BBF0A4311C6D3C657BE4D383E400BB9FD3D2240ED38FFC4C0429035E1B9BEC2B4C58EC8D5BB00BB1541C1C15AB3F63F3BBC2B2F963F62426D3D7946B73109C1F434CD3E58BFBA36EFACE1407A3E2BC4C5C23FBC354856C5B8C440C13C4144C4204249402AC12241"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[2.961430e-01, -1.075440e-01, 1.652340e+00, -1.509770e+00, -4.233400e-01, 5.250000e+00, -2.697270e+00], [-1.001950e+00, 3.515630e+00, -2.300780e+00, 3.019530e+00, -4.963380e-01, 2.143860e-02, -3.469240e-01]]> : tensor<2x7xf16>}> : () -> tensor<2x7xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<2x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x213F40B019C44F43ACC41739E72CBD34DF427740253BC6B6404509421CC14EC4DAB82D4145C71992D6431AC88D2B5CB542B98F37F33DCBBAA9416645E4414234F2C0754558B53935FCC156C2DCB2FEC152B6D84215C6283D26422E2A1D427545AD4081C3BC321E443D3C9FAFE545D9BF26B818B8994308450238ABB2B4C6843E263554B4A3B4EE3C2F3D79C2FF46794401453FC0D5C284C226C058C2D4BA62B71F40E9B7CA457DBD773CE63E3C380831663DDFC0F7C0F0C20FC5873C2F4188BABFBB08C058B9734087A9F9464DC537C6CA3C843E0843FDBC0A42F1B77D258DB57CC30F3F714175BD014161C2083641442FC11BA8133EAD32B33D53BAD7BE0C312937E2C5DABA2C4173BD01C451ACE8445A402BB676BF01C5BD44FBB6EB364D42453CF5BDBB3D5243FF4135BDCA4832B6DA40B1B4F03B48BE914170C21BBF0A4311C6D3C657BE4D383E400BB9FD3D2240ED38FFC4C0429035E1B9BEC2B4C58EC8D5BB00BB1541C1C15AB3F63F3BBC2B2F963F62426D3D7946B73109C1F434CD3E58BFBA36EFACE1407A3E2BC4C5C23FBC354856C5B8C440C13C4144C4204249402AC12241"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

