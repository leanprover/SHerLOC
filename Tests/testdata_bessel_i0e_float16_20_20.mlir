"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf16>
    %4 = "stablehlo.convert"(%2) : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %5 = "stablehlo.abs"(%4) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %6 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %8 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %10 = "stablehlo.constant"() <{value = dense<3.200000e+01> : tensor<f32>}> : () -> tensor<f32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %12 = "stablehlo.multiply"(%7, %5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %13 = "stablehlo.subtract"(%12, %9) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %15 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %17 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %18 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %19 = "stablehlo.broadcast_in_dim"(%18) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %20 = "stablehlo.multiply"(%13, %15) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %21 = "stablehlo.subtract"(%20, %17) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %22 = "stablehlo.constant"() <{value = dense<-1.300025009986248E-8> : tensor<f64>}> : () -> tensor<f64>
    %23 = "stablehlo.convert"(%22) : (tensor<f64>) -> tensor<f32>
    %24 = "stablehlo.broadcast_in_dim"(%23) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %25 = "stablehlo.add"(%21, %24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %26 = "stablehlo.multiply"(%13, %25) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %27 = "stablehlo.subtract"(%26, %15) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %28 = "stablehlo.constant"() <{value = dense<6.0469950225419186E-8> : tensor<f64>}> : () -> tensor<f64>
    %29 = "stablehlo.convert"(%28) : (tensor<f64>) -> tensor<f32>
    %30 = "stablehlo.broadcast_in_dim"(%29) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %31 = "stablehlo.add"(%27, %30) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "stablehlo.multiply"(%13, %31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "stablehlo.subtract"(%32, %25) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %34 = "stablehlo.constant"() <{value = dense<-2.6707938539406119E-7> : tensor<f64>}> : () -> tensor<f64>
    %35 = "stablehlo.convert"(%34) : (tensor<f64>) -> tensor<f32>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %37 = "stablehlo.add"(%33, %36) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %38 = "stablehlo.multiply"(%13, %37) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %39 = "stablehlo.subtract"(%38, %31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %40 = "stablehlo.constant"() <{value = dense<1.1173875391201037E-6> : tensor<f64>}> : () -> tensor<f64>
    %41 = "stablehlo.convert"(%40) : (tensor<f64>) -> tensor<f32>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %43 = "stablehlo.add"(%39, %42) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %44 = "stablehlo.multiply"(%13, %43) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %45 = "stablehlo.subtract"(%44, %37) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %46 = "stablehlo.constant"() <{value = dense<-4.4167383584587505E-6> : tensor<f64>}> : () -> tensor<f64>
    %47 = "stablehlo.convert"(%46) : (tensor<f64>) -> tensor<f32>
    %48 = "stablehlo.broadcast_in_dim"(%47) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %49 = "stablehlo.add"(%45, %48) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.multiply"(%13, %49) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %51 = "stablehlo.subtract"(%50, %43) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %52 = "stablehlo.constant"() <{value = dense<1.6448448070728896E-5> : tensor<f64>}> : () -> tensor<f64>
    %53 = "stablehlo.convert"(%52) : (tensor<f64>) -> tensor<f32>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %55 = "stablehlo.add"(%51, %54) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %56 = "stablehlo.multiply"(%13, %55) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %57 = "stablehlo.subtract"(%56, %49) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.constant"() <{value = dense<-5.754195010082104E-5> : tensor<f64>}> : () -> tensor<f64>
    %59 = "stablehlo.convert"(%58) : (tensor<f64>) -> tensor<f32>
    %60 = "stablehlo.broadcast_in_dim"(%59) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %61 = "stablehlo.add"(%57, %60) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %62 = "stablehlo.multiply"(%13, %61) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %63 = "stablehlo.subtract"(%62, %55) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %64 = "stablehlo.constant"() <{value = dense<1.8850288509584165E-4> : tensor<f64>}> : () -> tensor<f64>
    %65 = "stablehlo.convert"(%64) : (tensor<f64>) -> tensor<f32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %67 = "stablehlo.add"(%63, %66) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %68 = "stablehlo.multiply"(%13, %67) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.subtract"(%68, %61) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.constant"() <{value = dense<-5.7637557453858236E-4> : tensor<f64>}> : () -> tensor<f64>
    %71 = "stablehlo.convert"(%70) : (tensor<f64>) -> tensor<f32>
    %72 = "stablehlo.broadcast_in_dim"(%71) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %73 = "stablehlo.add"(%69, %72) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %74 = "stablehlo.multiply"(%13, %73) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %75 = "stablehlo.subtract"(%74, %67) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %76 = "stablehlo.constant"() <{value = dense<0.0016394756169413357> : tensor<f64>}> : () -> tensor<f64>
    %77 = "stablehlo.convert"(%76) : (tensor<f64>) -> tensor<f32>
    %78 = "stablehlo.broadcast_in_dim"(%77) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %79 = "stablehlo.add"(%75, %78) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %80 = "stablehlo.multiply"(%13, %79) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %81 = "stablehlo.subtract"(%80, %73) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %82 = "stablehlo.constant"() <{value = dense<-0.0043243099950505759> : tensor<f64>}> : () -> tensor<f64>
    %83 = "stablehlo.convert"(%82) : (tensor<f64>) -> tensor<f32>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %85 = "stablehlo.add"(%81, %84) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %86 = "stablehlo.multiply"(%13, %85) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %87 = "stablehlo.subtract"(%86, %79) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %88 = "stablehlo.constant"() <{value = dense<0.010546460394594998> : tensor<f64>}> : () -> tensor<f64>
    %89 = "stablehlo.convert"(%88) : (tensor<f64>) -> tensor<f32>
    %90 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %91 = "stablehlo.add"(%87, %90) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %92 = "stablehlo.multiply"(%13, %91) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %93 = "stablehlo.subtract"(%92, %85) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %94 = "stablehlo.constant"() <{value = dense<-0.023737414805899471> : tensor<f64>}> : () -> tensor<f64>
    %95 = "stablehlo.convert"(%94) : (tensor<f64>) -> tensor<f32>
    %96 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %97 = "stablehlo.add"(%93, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %98 = "stablehlo.multiply"(%13, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %99 = "stablehlo.subtract"(%98, %91) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %100 = "stablehlo.constant"() <{value = dense<0.049305284239670712> : tensor<f64>}> : () -> tensor<f64>
    %101 = "stablehlo.convert"(%100) : (tensor<f64>) -> tensor<f32>
    %102 = "stablehlo.broadcast_in_dim"(%101) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %103 = "stablehlo.add"(%99, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %104 = "stablehlo.multiply"(%13, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %105 = "stablehlo.subtract"(%104, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %106 = "stablehlo.constant"() <{value = dense<-0.094901097048047639> : tensor<f64>}> : () -> tensor<f64>
    %107 = "stablehlo.convert"(%106) : (tensor<f64>) -> tensor<f32>
    %108 = "stablehlo.broadcast_in_dim"(%107) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %109 = "stablehlo.add"(%105, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %110 = "stablehlo.multiply"(%13, %109) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %111 = "stablehlo.subtract"(%110, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %112 = "stablehlo.constant"() <{value = dense<0.17162090152220877> : tensor<f64>}> : () -> tensor<f64>
    %113 = "stablehlo.convert"(%112) : (tensor<f64>) -> tensor<f32>
    %114 = "stablehlo.broadcast_in_dim"(%113) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %115 = "stablehlo.add"(%111, %114) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %116 = "stablehlo.multiply"(%13, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %117 = "stablehlo.subtract"(%116, %109) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %118 = "stablehlo.constant"() <{value = dense<-0.3046826723431984> : tensor<f64>}> : () -> tensor<f64>
    %119 = "stablehlo.convert"(%118) : (tensor<f64>) -> tensor<f32>
    %120 = "stablehlo.broadcast_in_dim"(%119) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %121 = "stablehlo.add"(%117, %120) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %122 = "stablehlo.multiply"(%13, %121) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %123 = "stablehlo.subtract"(%122, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %124 = "stablehlo.constant"() <{value = dense<0.67679527440947607> : tensor<f64>}> : () -> tensor<f64>
    %125 = "stablehlo.convert"(%124) : (tensor<f64>) -> tensor<f32>
    %126 = "stablehlo.broadcast_in_dim"(%125) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %127 = "stablehlo.add"(%123, %126) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %128 = "stablehlo.subtract"(%127, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %129 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %130 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %131 = "stablehlo.multiply"(%130, %128) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %132 = "stablehlo.divide"(%11, %5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %133 = "stablehlo.subtract"(%132, %9) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %134 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %135 = "stablehlo.broadcast_in_dim"(%134) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %136 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %137 = "stablehlo.broadcast_in_dim"(%136) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %138 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %139 = "stablehlo.broadcast_in_dim"(%138) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %140 = "stablehlo.multiply"(%133, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %141 = "stablehlo.subtract"(%140, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %142 = "stablehlo.constant"() <{value = dense<3.3962320257083865E-9> : tensor<f64>}> : () -> tensor<f64>
    %143 = "stablehlo.convert"(%142) : (tensor<f64>) -> tensor<f32>
    %144 = "stablehlo.broadcast_in_dim"(%143) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %145 = "stablehlo.add"(%141, %144) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %146 = "stablehlo.multiply"(%133, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %147 = "stablehlo.subtract"(%146, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %148 = "stablehlo.constant"() <{value = dense<2.266668990498178E-8> : tensor<f64>}> : () -> tensor<f64>
    %149 = "stablehlo.convert"(%148) : (tensor<f64>) -> tensor<f32>
    %150 = "stablehlo.broadcast_in_dim"(%149) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %151 = "stablehlo.add"(%147, %150) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %152 = "stablehlo.multiply"(%133, %151) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %153 = "stablehlo.subtract"(%152, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %154 = "stablehlo.constant"() <{value = dense<2.0489185894690638E-7> : tensor<f64>}> : () -> tensor<f64>
    %155 = "stablehlo.convert"(%154) : (tensor<f64>) -> tensor<f32>
    %156 = "stablehlo.broadcast_in_dim"(%155) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %157 = "stablehlo.add"(%153, %156) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %158 = "stablehlo.multiply"(%133, %157) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %159 = "stablehlo.subtract"(%158, %151) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %160 = "stablehlo.constant"() <{value = dense<2.8913705208347567E-6> : tensor<f64>}> : () -> tensor<f64>
    %161 = "stablehlo.convert"(%160) : (tensor<f64>) -> tensor<f32>
    %162 = "stablehlo.broadcast_in_dim"(%161) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %163 = "stablehlo.add"(%159, %162) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %164 = "stablehlo.multiply"(%133, %163) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %165 = "stablehlo.subtract"(%164, %157) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %166 = "stablehlo.constant"() <{value = dense<6.8897583469168245E-5> : tensor<f64>}> : () -> tensor<f64>
    %167 = "stablehlo.convert"(%166) : (tensor<f64>) -> tensor<f32>
    %168 = "stablehlo.broadcast_in_dim"(%167) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %169 = "stablehlo.add"(%165, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %170 = "stablehlo.multiply"(%133, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %171 = "stablehlo.subtract"(%170, %163) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %172 = "stablehlo.constant"() <{value = dense<0.0033691164782556943> : tensor<f64>}> : () -> tensor<f64>
    %173 = "stablehlo.convert"(%172) : (tensor<f64>) -> tensor<f32>
    %174 = "stablehlo.broadcast_in_dim"(%173) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %175 = "stablehlo.add"(%171, %174) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %176 = "stablehlo.multiply"(%133, %175) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %177 = "stablehlo.subtract"(%176, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %178 = "stablehlo.constant"() <{value = dense<0.80449041101410879> : tensor<f64>}> : () -> tensor<f64>
    %179 = "stablehlo.convert"(%178) : (tensor<f64>) -> tensor<f32>
    %180 = "stablehlo.broadcast_in_dim"(%179) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %181 = "stablehlo.add"(%177, %180) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %182 = "stablehlo.subtract"(%181, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %183 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %184 = "stablehlo.broadcast_in_dim"(%183) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %185 = "stablehlo.multiply"(%184, %182) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %186 = "stablehlo.sqrt"(%5) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %187 = "stablehlo.divide"(%185, %186) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %188 = "stablehlo.constant"() <{value = dense<8.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %189 = "stablehlo.broadcast_in_dim"(%188) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %190 = "stablehlo.compare"(%5, %189) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %191 = "stablehlo.select"(%190, %131, %187) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %192 = "stablehlo.convert"(%191) : (tensor<20x20xf32>) -> tensor<20x20xf16>
    "stablehlo.custom_call"(%192, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    "func.return"(%192) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x913768BC07BE0D38E6B8D8BDFAC403B078B859441FC21BBCDBBC7141DB40184276C7E9C69BBB3B3985BB45407F3F313CA244BABA1943FABDEBC0C8BC20B5CD4314C4C1C79344E0C2E9BF572925C2503D7EBEABC16C39F7C307C0DCC4DAB9003D54C3173A2C3ACBC46F40C8BC7FAAC442C63C5C427531E9C635397939724187B5B1BCFC4345BD8CBAA83D0038254525BF9DBAEF3BE5C05B28EC421C410DC4183F95C7ED4211390DA9B5BC394068C3C1404EB8A64162A9C140243E7D442446EA43053EC43F5F39D845D0C4A543D442692A92C309440D3EB7C493BD663BC1A38344853D0E46C635D9C230409239E13CB4C41A4571C04FC4B230E6B608C6583CCAC1722C0CC2F7B5ACC137C074B863BCABC396BC6F387C3C4FBC4CC0233C4AC2FBBF3FC46840C243B8BC663C24C071B9423CFF36A2C2294628AC4E3FBD36AEC6CDBC57B3D6BE3FC2EF39B53053C03042D5C068BC05B9863BD9406542A43927C146C4ACB6DC344C46DABA5D3D2D40B144213DCCB13BB71DBC9141F8BF6F4668C4DD410F4326BEEBB6E0C08144293F2EC588BBB6BF792F4F38E1BCFD4076455CACE8B4B63C9F401EC098C4F541A1C5AA40AAB5F1BD9FB5F23C75BE59BB3DB8923C4F4038BDBB3895BAC43A83C0CBC3B534A929743EE4C3CA2791C41EBFF544CCC3543F25BE21420E42263BD7386B3CC24314C4804158B9CE40DE3908C53EC2E4C4B93F014068BF05BFFBC1F14518C105C09148A3B5F24702BD1BC45F41A5366E3CF7BA5C38033F75B8E839FBBE3C3EA1C0304030BDA33DCF446B44383AC8413640CCBC4B398F398FBCF93CE0C4B2BE04446FBADCC1B83A1E425B3E55C299440AB53E4380C7A1BF8248BDC269B862BF68B122427C3CE4C41945B941B64168C4F0C40838CB2800BFDB34F2C41BBE8CBB18C416391533B641B54546C344BCEEC1B03BAE415142BFBC3E2885C4C4C2AB40B73A5BC2154517BD4FC86F3898B81FC47F4099410F4467BDAE4467340D46FCBB764036BFA0402A4063C5E63DC73B053E3C4503C6163CD0399EB8A0BD4F3D23C7CBBC1B3DC5C55EC3403F3FC2D6B88E443E447847B5C1E02BB841544176B1EAC0D2438FC00938FDC5E4C0C1B4F03E02C465C04342"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%1) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x45390F37DD352339C138F935E331163BF1385532B0335837AC361D346534B533C230F430A9379E38B637BF34223543371E3217381333E5355D34BB36F539B7328E32AA3029323433F834AD3BAC3353369B3504348B38A432EA34F73162388F36F3324C3845380232A434BB369C3B4433BD368633CB3AF430A03886381C34D539CF36A2325B36253818362939C931493520387D376034BC3B2D33443494324F35B8302C33AF38B23BCB36C734E932743404390634AD3B7434CC3539324531AA32DE35073590386831FF31CA323B339D3BD4329832D93510322536C737E13B35322F364F31C2393833CD347C38A7361232D031A3345D32F23A713952311D37F033793BBE33B4390434C834F2381337C832E636F538FD362637BA3450379233F2346A32A834BD32C9361137D634893832376B3959334231813B37357C390B31B736723A6E359A335A38F13AB634A43369340F37B438B5376634803376383F34643281390B3A33310D384A36CF3414327636BB3A5B3956370F34F33424314932E1331933CB357039633436324835C431B4370C35253B0339A73654349B317B3B073ACA368734DA342532CF3384318034CB39EA35CE399A369F35CE370C39E936B8346536D338223813389734B832173AA83BA035AD32C33B2A324D35E731B8323535CB35AF33BC33EC37C7380C37BD328E32163493386C346038DB319A33F1310B35EF342C355835CA335C314634EC344930CD399B308D3688322534833909370438FD385935F2385D385D35BE358634CD346A361B36FF3147324138F133C934B83698387D38EC369436F43180359C322F38E2331738B133AD358B332532FC39FF32BF30153551304833F7382F35CE3AAE33FD36F131D031FD3300344932EA312539B53B5A350B3AE831D135B2378A32AD387E3A00347A31FB323037D4339E3703348D33C336BE3B33324433803418388733D3317D366B30F538E23884329A340C34923243361632313A4F3176379F3442358634D134A531F1359237DE35BB3154315D376538E0381D365436DF30B9367A367231EE323D359A33C8382D326A32C2300034883BFE332A34CB3A5D34B5329034253957316134133A62359E32AA349733"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%0) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

